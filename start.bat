@echo off
echo ======================================================================
echo          WORKFLOW GENERATOR - STARTUP SCRIPT
echo ======================================================================
echo.
echo Starting the AI-powered Workflow Generator...
echo.
echo This will:
echo   1. Start the MCP backend server
echo   2. Launch the REST API server
echo   3. Serve the premium HTML frontend
echo.
echo Once started, open your browser to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ======================================================================
echo.

python server\api_server.py

pause
