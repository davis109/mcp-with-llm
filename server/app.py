"""
Unified Workflow Generator Server
===================================
Combines REST API server and MCP workflow engine into a single application.
Provides HTTP endpoints for the frontend and internal MCP-based workflow generation.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import json
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional
from enum import Enum
from dotenv import load_dotenv

from pydantic import BaseModel, Field, validator

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv('GEMINI_API_KEY')
print(f"🔑 ENV PATH: {env_path}", file=sys.stderr)
print(f"🔑 ENV EXISTS: {env_path.exists()}", file=sys.stderr)
print(f"🔑 GEMINI_API_KEY loaded: {'YES - ' + api_key[:20] + '...' if api_key else 'NO'}", file=sys.stderr)

# Try to import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Using fallback NL processing.", file=sys.stderr)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Path configuration
WORKFLOWS_DIR = Path(__file__).parent.parent / "client"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "data.json"

# ============================================================================
# WORKFLOW ENGINE MODELS AND LOGIC (from server.py)
# ============================================================================

class WorkflowStepType(str, Enum):
    MANUAL_TASK = "manual_task"
    AUTOMATED_TASK = "automated_task"
    DECISION_GATE = "decision_gate"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    DATA_TRANSFORMATION = "data_transformation"
    INTEGRATION = "integration"
    TIMER = "timer"


class WorkflowStep(BaseModel):
    id: str = Field(..., description="Unique identifier for the step")
    type: str = Field(..., description="Type of the step (e.g., manual_task, approval)")
    name: str = Field(..., description="Human-readable name of the step")
    description: Optional[str] = Field(None, description="Detailed description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Step-specific properties")
    assigned_role: Optional[str] = Field(None, description="Role assigned to execute this step")


class WorkflowConnection(BaseModel):
    from_step: str = Field(..., description="ID of the source step")
    to_step: str = Field(..., description="ID of the destination step")
    condition: Optional[str] = Field(None, description="Conditional logic for this connection")
    label: Optional[str] = Field(None, description="Label for the connection")


class WorkflowMetadata(BaseModel):
    name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of what the workflow does")
    domain: str = Field(..., description="Business domain (e.g., finance, hr, it)")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = Field(default="1.0.0", description="Workflow version")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")


class WorkflowBlueprint(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow identifier")
    metadata: WorkflowMetadata
    steps: List[WorkflowStep] = Field(..., description="List of workflow steps")
    connections: List[WorkflowConnection] = Field(..., description="Connections between steps")
    start_step: str = Field(..., description="ID of the starting step")
    end_steps: List[str] = Field(..., description="IDs of terminal steps")

    @validator('steps')
    def validate_steps_not_empty(cls, v):
        if not v:
            raise ValueError("Workflow must have at least one step")
        return v

    @validator('start_step')
    def validate_start_step_exists(cls, v, values):
        if 'steps' in values:
            step_ids = {step.id for step in values['steps']}
            if v not in step_ids:
                raise ValueError(f"start_step '{v}' not found in steps")
        return v


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEngine:
    """Core workflow generation and management engine"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_data = self._load_config()
        self.workflows: Dict[str, WorkflowBlueprint] = {}

        self.gemini_model = None
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(model_name)
                    print(f"✅ Gemini AI enabled: {model_name}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  Gemini initialization failed: {e}", file=sys.stderr)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in configuration file: {e}")

    def get_step_types(self) -> List[Dict[str, Any]]:
        return self.config_data.get("step_types", [])

    def get_roles(self) -> List[Dict[str, Any]]:
        return self.config_data.get("roles", [])

    def generate_workflow(self, description: str, domain: str) -> WorkflowBlueprint:
        workflow_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        analysis = self._analyze_description(description, domain)
        steps = self._build_steps(analysis, domain)
        connections = self._build_connections(steps, analysis)
        start_step = steps[0].id if steps else None
        end_steps = [steps[-1].id] if steps else []

        metadata = WorkflowMetadata(
            name=analysis.get("workflow_name", f"{domain.title()} Workflow"),
            description=description,
            domain=domain,
            tags=analysis.get("tags", [domain])
        )

        blueprint = WorkflowBlueprint(
            workflow_id=workflow_id,
            metadata=metadata,
            steps=steps,
            connections=connections,
            start_step=start_step,
            end_steps=end_steps
        )

        self.workflows[workflow_id] = blueprint
        return blueprint

    def _analyze_description(self, description: str, domain: str) -> Dict[str, Any]:
        if self.gemini_model:
            try:
                print(f"\n🔍 Using GEMINI AI for workflow analysis", file=sys.stderr)
                return self._analyze_with_gemini(description, domain)
            except Exception as e:
                print(f"⚠️  Gemini analysis failed, using fallback: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"\n⚠️  Gemini not available - using FALLBACK keyword matching", file=sys.stderr)

        desc_lower = description.lower()
        analysis = {
            "workflow_name": "Generated Workflow",
            "tags": [domain],
            "detected_steps": [],
            "detected_roles": []
        }

        step_keywords = {
            "approval": ["approve", "approval", "authorize", "sign off"],
            "manual_task": ["task", "manual", "perform", "execute", "complete"],
            "automated_task": ["automate", "script", "system", "automatic"],
            "notification": ["notify", "email", "alert", "message", "send"],
            "decision_gate": ["if", "decide", "choose", "branch", "condition"],
            "data_transformation": ["transform", "convert", "process data", "map"],
            "integration": ["integrate", "api", "connect", "external system"],
            "timer": ["wait", "delay", "schedule", "after"]
        }

        for step_type, keywords in step_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis["detected_steps"].append(step_type)

        role_keywords = {
            "approver": ["approve", "manager", "supervisor"],
            "executor": ["execute", "worker", "team member"],
            "reviewer": ["review", "check", "verify"],
            "administrator": ["admin", "configure"],
            "system": ["automatic", "system", "bot"]
        }

        for role, keywords in role_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis["detected_roles"].append(role)

        if not analysis["detected_steps"]:
            domain_template = self.config_data.get("domain_templates", {}).get(domain, {})
            analysis["detected_steps"] = domain_template.get("common_steps", ["manual_task"])

        if not analysis["detected_roles"]:
            analysis["detected_roles"] = ["executor", "approver"]

        return analysis

    def _analyze_with_gemini(self, description: str, domain: str) -> Dict[str, Any]:
        print(f"\n🤖 GEMINI AI ANALYZING:", file=sys.stderr)
        print(f"   Description: {description[:100]}...", file=sys.stderr)
        print(f"   Domain: {domain}", file=sys.stderr)

        prompt = f"""You are a workflow expert. Analyze this business process and create a DETAILED, REALISTIC workflow with specific steps.

Business Process: {description}
Domain: {domain}

Available step types: manual_task, automated_task, decision_gate, approval, notification, data_transformation, integration, timer
Available roles: initiator, approver, executor, reviewer, administrator, system

Create a workflow with 5-10 SPECIFIC steps that match the business process. Each step should have:
- A descriptive name relevant to the actual process (NOT generic like "Manual Task")
- A specific step type from the available types
- An appropriate role
- Properties object with relevant configuration (e.g., approvers, timeout, conditions)

For DECISION_GATE steps, include multiple possible outcomes in a "branches" array.
For APPROVAL steps, include "approvers" list and "timeout_hours" in properties.
For TIMER steps, include "duration" and "unit" in properties.

Example for "Bug Reporting Workflow":
{{
  "workflow_name": "Bug Reporting and Resolution",
  "tags": ["customer_service", "bug_tracking", "development"],
  "steps": [
    {{"name": "Submit Bug Report", "type": "manual_task", "description": "User submits bug details with screenshots", "role": "initiator", "properties": {{"required_fields": ["description", "steps_to_reproduce"]}}}},
    {{"name": "Validate Bug Information", "type": "manual_task", "description": "Support team verifies bug details are complete", "role": "reviewer", "properties": {{}}}},
    {{"name": "Assess Priority", "type": "decision_gate", "description": "Determine urgency: Critical, High, Medium, Low", "role": "reviewer", "properties": {{}}, "branches": ["critical", "high", "medium", "low"]}},
    {{"name": "Assign to Developer", "type": "automated_task", "description": "Auto-assign based on priority and team availability", "role": "system", "properties": {{"assignment_rules": "priority_based"}}}},
    {{"name": "Notify Development Team", "type": "notification", "description": "Send email/Slack to assigned developer", "role": "system", "properties": {{"channels": ["email", "slack"]}}}},
    {{"name": "Fix and Test Bug", "type": "manual_task", "description": "Developer fixes issue and runs tests", "role": "executor", "properties": {{}}}},
    {{"name": "Code Review", "type": "approval", "description": "Senior developer reviews the fix", "role": "approver", "properties": {{"approvers": ["senior_developer"], "timeout_hours": 24}}}},
    {{"name": "Deploy to Production", "type": "integration", "description": "Automated deployment via CI/CD", "role": "system", "properties": {{"deployment_type": "blue_green"}}}},
    {{"name": "Notify Customer", "type": "notification", "description": "Email customer that bug is fixed", "role": "system", "properties": {{"channels": ["email"]}}}}
  ]
}}

NOW analyze "{description}" in the {domain} domain and create a similar DETAILED workflow.
Return ONLY valid JSON with workflow_name, tags, and steps array. Each step must have: name, type, description, role, properties. Add "branches" for decision_gate steps."""

        print(f"   Calling Gemini API...", file=sys.stderr)
        import time
        start_time = time.time()

        response = self.gemini_model.generate_content(prompt)
        result_text = response.text.strip()

        elapsed = time.time() - start_time
        print(f"   ✅ Gemini responded in {elapsed:.2f}s", file=sys.stderr)
        print(f"   Response length: {len(result_text)} chars", file=sys.stderr)

        if result_text.startswith('```'):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else lines[1]

        print(f"   📄 Parsing JSON response...", file=sys.stderr)
        print(f"   First 200 chars: {result_text[:200]}", file=sys.stderr)

        try:
            analysis = json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON PARSE ERROR: {e}", file=sys.stderr)
            print(f"   Full response:\n{result_text}", file=sys.stderr)
            raise

        print(f"   ✅ Parsed successfully! Keys: {list(analysis.keys())}", file=sys.stderr)

        if 'workflow_name' not in analysis:
            analysis['workflow_name'] = f"{domain.title()} Workflow"
        if 'tags' not in analysis:
            analysis['tags'] = [domain]
        if 'steps' not in analysis and 'detected_steps' not in analysis:
            analysis['detected_steps'] = ['manual_task', 'approval']
        if 'detected_roles' not in analysis:
            analysis['detected_roles'] = ['executor', 'approver']

        return analysis

    def _build_steps(self, analysis: Dict[str, Any], domain: str) -> List[WorkflowStep]:
        steps = []
        gemini_steps = analysis.get("steps", [])

        if gemini_steps:
            for idx, step_data in enumerate(gemini_steps):
                steps.append(WorkflowStep(
                    id=f"step_{idx:03d}" if idx > 0 else "step_start",
                    type=step_data.get("type", "manual_task"),
                    name=step_data.get("name", f"Step {idx + 1}"),
                    description=step_data.get("description", ""),
                    properties=step_data.get("properties", {}),
                    assigned_role=step_data.get("role", "executor")
                ))
        else:
            detected_step_types = analysis.get("detected_steps", [])

            steps.append(WorkflowStep(
                id="step_start",
                type="manual_task",
                name="Initiate Workflow",
                description="Start the workflow process",
                assigned_role="initiator"
            ))

            for idx, step_type in enumerate(detected_step_types, start=1):
                step_config = next(
                    (s for s in self.config_data["step_types"] if s["id"] == step_type),
                    None
                )

                if step_config:
                    steps.append(WorkflowStep(
                        id=f"step_{idx:03d}",
                        type=step_type,
                        name=step_config["name"],
                        description=step_config["description"],
                        properties={},
                        assigned_role=analysis.get("detected_roles", ["executor"])[0] if analysis.get("detected_roles") else "executor"
                    ))

            steps.append(WorkflowStep(
                id="step_end",
                type="notification",
                name="Workflow Complete",
                description="Notify stakeholders of completion",
                assigned_role="system"
            ))

        return steps

    def _build_connections(self, steps: List[WorkflowStep], analysis: Dict[str, Any]) -> List[WorkflowConnection]:
        connections = []
        gemini_steps = analysis.get("steps", [])

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            if current_step.type == "decision_gate" and i < len(gemini_steps):
                gemini_step = gemini_steps[i]
                branches = gemini_step.get("branches", [])

                if branches and len(branches) >= 2:
                    # Handle both string and dict branches
                    first_branch = branches[0]
                    second_branch = branches[1]
                    
                    # Convert dict to string if needed
                    if isinstance(first_branch, dict):
                        first_branch = first_branch.get("name") or first_branch.get("condition") or str(first_branch)
                    if isinstance(second_branch, dict):
                        second_branch = second_branch.get("name") or second_branch.get("condition") or str(second_branch)

                    skip_keywords = ["rejected", "denied", "no", "failed", "ineligible",
                                     "proceed_to_payment", "skip", "bypass", "eligible_for_next",
                                     "backlog", "future", "postpone", "defer", "later"]

                    continue_keywords = ["approved", "yes", "requires", "needs", "manual",
                                         "review", "high", "escalate", "current", "planned", "immediate"]

                    first_should_skip = any(kw in str(first_branch).lower() for kw in skip_keywords)
                    second_should_skip = any(kw in str(second_branch).lower() for kw in skip_keywords)

                    if first_should_skip:
                        if i + 2 < len(steps):
                            connections.append(WorkflowConnection(
                                from_step=current_step.id,
                                to_step=steps[i + 2].id,
                                condition=first_branch,
                                label=f"If {first_branch}"
                            ))
                        else:
                            connections.append(WorkflowConnection(
                                from_step=current_step.id,
                                to_step=steps[-1].id,
                                condition=first_branch,
                                label=f"If {first_branch}"
                            ))
                    else:
                        connections.append(WorkflowConnection(
                            from_step=current_step.id,
                            to_step=next_step.id,
                            condition=first_branch,
                            label=f"If {first_branch}"
                        ))

                    if second_should_skip:
                        if i + 2 < len(steps):
                            connections.append(WorkflowConnection(
                                from_step=current_step.id,
                                to_step=steps[i + 2].id,
                                condition=second_branch,
                                label=f"If {second_branch}"
                            ))
                        else:
                            connections.append(WorkflowConnection(
                                from_step=current_step.id,
                                to_step=steps[-1].id,
                                condition=second_branch,
                                label=f"If {second_branch}"
                            ))
                    else:
                        connections.append(WorkflowConnection(
                            from_step=current_step.id,
                            to_step=next_step.id,
                            condition=second_branch,
                            label=f"If {second_branch}"
                        ))
                else:
                    connections.append(WorkflowConnection(
                        from_step=current_step.id,
                        to_step=next_step.id,
                        label=f"Proceed to {next_step.name}"
                    ))
            else:
                connections.append(WorkflowConnection(
                    from_step=current_step.id,
                    to_step=next_step.id,
                    label=f"Proceed to {next_step.name}"
                ))

        return connections

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        statistics = {}

        try:
            blueprint = WorkflowBlueprint(**workflow_data)

            step_ids = {step.id for step in blueprint.steps}
            statistics["total_steps"] = len(blueprint.steps)
            statistics["total_connections"] = len(blueprint.connections)

            if blueprint.start_step not in step_ids:
                errors.append(f"Start step '{blueprint.start_step}' does not exist")

            for end_step in blueprint.end_steps:
                if end_step not in step_ids:
                    errors.append(f"End step '{end_step}' does not exist")

            for conn in blueprint.connections:
                if conn.from_step not in step_ids:
                    errors.append(f"Connection references non-existent from_step: {conn.from_step}")
                if conn.to_step not in step_ids:
                    errors.append(f"Connection references non-existent to_step: {conn.to_step}")

            connected_steps = set()
            connected_steps.add(blueprint.start_step)
            for conn in blueprint.connections:
                connected_steps.add(conn.from_step)
                connected_steps.add(conn.to_step)

            orphaned = step_ids - connected_steps
            if orphaned:
                warnings.append(f"Orphaned steps detected: {', '.join(orphaned)}")

            if self._has_cycles(blueprint):
                warnings.append("Potential circular dependencies detected")

            statistics["orphaned_steps"] = len(orphaned)

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

    def _has_cycles(self, blueprint: WorkflowBlueprint) -> bool:
        graph = {}
        for conn in blueprint.connections:
            if conn.from_step not in graph:
                graph[conn.from_step] = []
            graph[conn.from_step].append(conn.to_step)

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step in blueprint.steps:
            if step.id not in visited:
                if dfs(step.id):
                    return True

        return False

    def export_to_format(self, workflow_id: str, format: Literal['json', 'yaml', 'bpmn']) -> str:
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        blueprint = self.workflows[workflow_id]

        if format == 'json':
            return json.dumps(blueprint.model_dump(), indent=2)

        elif format == 'yaml':
            return yaml.dump(blueprint.model_dump(), default_flow_style=False, sort_keys=False)

        elif format == 'bpmn':
            return self._export_to_bpmn(blueprint)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_to_bpmn(self, blueprint: WorkflowBlueprint) -> str:
        bpmn = f'''<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
             id="definitions_{blueprint.workflow_id}"
             targetNamespace="http://workflow.generator/bpmn">
  
  <process id="{blueprint.workflow_id}" name="{blueprint.metadata.name}" isExecutable="true">
    
    <startEvent id="start_{blueprint.start_step}" name="Start">
      <outgoing>flow_start</outgoing>
    </startEvent>
    
'''

        for step in blueprint.steps:
            is_user_task = (
                "manual" in step.type or 
                "approval" in step.type or 
                "reviewer" in step.assigned_role.lower() or
                "manager" in step.assigned_role.lower() or
                "approver" in step.assigned_role.lower() or
                "creator" in step.assigned_role.lower() or
                "counsel" in step.assigned_role.lower() or
                "director" in step.assigned_role.lower() or
                "officer" in step.assigned_role.lower() or
                "sme" in step.assigned_role.lower() or
                "specialist" in step.assigned_role.lower()
            )
            
            task_type = "userTask" if is_user_task else "serviceTask"
            
            incoming = [f"flow_{idx}" for idx, conn in enumerate(blueprint.connections) if conn.to_step == step.id]
            outgoing = [f"flow_{idx}" for idx, conn in enumerate(blueprint.connections) if conn.from_step == step.id]
            
            bpmn += f'''    <{task_type} id="{step.id}" name="{step.name}">
      <documentation>{step.description or ''}</documentation>
'''
            for inc in incoming:
                bpmn += f'''      <incoming>{inc}</incoming>
'''
            for out in outgoing:
                bpmn += f'''      <outgoing>{out}</outgoing>
'''
            bpmn += f'''    </{task_type}>
    
'''

        gateways_added = set()
        for idx, conn in enumerate(blueprint.connections):
            if conn.condition:
                gateway_id = f"gateway_{conn.from_step}"
                if gateway_id not in gateways_added:
                    outgoing_flows = [f"flow_{i}" for i, c in enumerate(blueprint.connections) if c.from_step == conn.from_step]
                    incoming_flows = [f"flow_{i}" for i, c in enumerate(blueprint.connections) if c.to_step == conn.from_step]
                    
                    bpmn += f'''    <exclusiveGateway id="{gateway_id}" name="Decision">
'''
                    for inc in incoming_flows:
                        bpmn += f'''      <incoming>{inc}</incoming>
'''
                    for out in outgoing_flows:
                        bpmn += f'''      <outgoing>{out}</outgoing>
'''
                    bpmn += f'''    </exclusiveGateway>
    
'''
                    gateways_added.add(gateway_id)

        for end_step in blueprint.end_steps:
            incoming_flows = [f"flow_{idx}" for idx, conn in enumerate(blueprint.connections) if conn.to_step == end_step]
            bpmn += f'''    <endEvent id="end_{end_step}" name="End">
'''
            for inc in incoming_flows:
                bpmn += f'''      <incoming>{inc}</incoming>
'''
            bpmn += f'''    </endEvent>
    
'''

        bpmn += f'''    <sequenceFlow id="flow_start" sourceRef="start_{blueprint.start_step}" targetRef="{blueprint.start_step}"/>
'''
        
        for idx, conn in enumerate(blueprint.connections):
            condition_name = f' name="{conn.label}"' if conn.label else ''
            bpmn += f'''    <sequenceFlow id="flow_{idx}" sourceRef="{conn.from_step}" targetRef="{conn.to_step}"{condition_name}/>
'''

        bpmn += '''  </process>
  
</definitions>'''

        return bpmn


# ============================================================================
# FLASK API ROUTES
# ============================================================================

# Initialize the workflow engine
engine = None

def init_engine():
    """Initialize workflow engine on startup"""
    global engine
    try:
        if not CONFIG_PATH.exists():
            print(f"❌ Configuration file not found: {CONFIG_PATH}", file=sys.stderr)
            sys.exit(1)
        engine = WorkflowEngine(CONFIG_PATH)
        print(f"✅ Workflow engine initialized with config: {CONFIG_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to initialize workflow engine: {e}", file=sys.stderr)
        sys.exit(1)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'engine_ready': engine is not None,
        'gemini_available': GEMINI_AVAILABLE
    })


@app.route('/api/generate', methods=['POST'])
def generate_workflow():
    """Generate workflow from natural language description"""
    if not engine:
        return jsonify({
            'success': False,
            'error': 'Workflow engine not initialized'
        }), 503
    
    data = request.get_json()
    description = data.get('description', '')
    domain = data.get('domain', 'general')
    
    if not description:
        return jsonify({
            'success': False,
            'error': 'Description is required'
        }), 400
    
    try:
        print(f"\n🎯 [API] Generate workflow request:", file=sys.stderr)
        print(f"   Description: {description[:80]}...", file=sys.stderr)
        print(f"   Domain: {domain}", file=sys.stderr)
        
        blueprint = engine.generate_workflow(description, domain)
        
        return jsonify({
            'success': True,
            'workflow_id': blueprint.workflow_id,
            'workflow': blueprint.model_dump()
        })
        
    except Exception as e:
        print(f"❌ Error generating workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_workflow():
    """Validate a workflow blueprint"""
    if not engine:
        return jsonify({
            'success': False,
            'error': 'Workflow engine not initialized'
        }), 503
    
    workflow = request.get_json()
    
    try:
        result = engine.validate_workflow(workflow)
        
        return jsonify({
            'validation_result': result.model_dump()
        })
        
    except Exception as e:
        print(f"❌ Error validating workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export', methods=['POST'])
def export_workflow():
    """Export workflow to different formats"""
    if not engine:
        return jsonify({
            'success': False,
            'error': 'Workflow engine not initialized'
        }), 503
    
    data = request.get_json()
    workflow = data.get('workflow')
    format_type = data.get('format', 'json')
    
    try:
        # Direct export from JSON
        blueprint = WorkflowBlueprint(**workflow)
        if format_type == 'json':
            exported = json.dumps(blueprint.model_dump(), indent=2)
        elif format_type == 'yaml':
            exported = yaml.dump(blueprint.model_dump(), default_flow_style=False, sort_keys=False)
        elif format_type == 'bpmn':
            exported = engine._export_to_bpmn(blueprint)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return exported, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        print(f"❌ Error exporting workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Serve static files
@app.route('/')
def index():
    return send_from_directory(WORKFLOWS_DIR, 'premium.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WORKFLOWS_DIR, path)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 70)
    try:
        print("🚀 UNIFIED WORKFLOW GENERATOR SERVER")
    except UnicodeEncodeError:
        print("UNIFIED WORKFLOW GENERATOR SERVER")
    print("=" * 70)
    
    # Initialize workflow engine
    init_engine()
    
    print(f"\n📁 Serving frontend from: {WORKFLOWS_DIR.absolute()}")
    print(f"⚙️  Using config: {CONFIG_PATH.absolute()}")
    print(f"\n🌐 API server starting on http://localhost:5000")
    print(f"🔗 Open in browser: http://localhost:5000")
    print("\n✋ Press Ctrl+C to stop")
    print("=" * 70)
    
    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Shutting down...", file=sys.stderr)


if __name__ == "__main__":
    main()
