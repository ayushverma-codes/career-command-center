import os
from sqlalchemy import create_engine, text
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv

# Note: In a real environment, you will need the mcp package installed for this to work.
# If you haven't installed it, you can mock this or run `uv pip install mcp`
try:
    from mcp import MCPToolset, StreamableHTTPConnectionParams
    mcp_available = True
except ImportError:
    mcp_available = False

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# ==========================================
# FEATURE 1: JOB TRACKER & RECRUITERS
# ==========================================
def log_job_application(tool_context: ToolContext, company: str, role: str) -> str:
    """Persists a new job application to AlloyDB."""
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO jobs (company_name, role_position, status) VALUES (:c, :r, 'Applied')"), 
            {"c": company, "r": role}
        )
        conn.commit()
    return f"Persisted: Application for {role} at {company} is now in your tracker."

# ==========================================
# FEATURE 2: RESUME AUDITOR (In-DB Intelligence)
# ==========================================
def audit_resume_against_jd(tool_context: ToolContext, jd: str, resume: str) -> str:
    """Uses ai.if() to check if resume notes match a JD."""
    query = text("""
        SELECT ai.if(
            prompt => 'Compare this resume: "' || :r || '" to this JD: "' || :j || '". Suggest 3 specific edits.',
            model_id => 'gemini-3-flash-preview'
        )
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"r": resume, "j": jd}).fetchone()
    return f"Auditor Suggestion: {result[0]}"

# ==========================================
# FEATURE 3: DSA TRACKER (Vector Search)
# ==========================================
def suggest_dsa_problem(tool_context: ToolContext, mistake: str) -> str:
    """Finds logic-based DSA suggestions using vector similarity."""
    query = text("""
        SELECT title FROM dsa_problems 
        ORDER BY problem_vector <=> embedding('text-embedding-005', :m)::vector LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"m": mistake}).fetchone()
    return f"Based on your mistake, practice: {result[0]}" if result else "Keep practicing your weak areas!"

# ==========================================
# FEATURE 4: PROGRESS ANALYZER (Tool)
# ==========================================
def analyze_my_progress(tool_context: ToolContext) -> str:
    """Synthesizes application status and mistakes into a raw report."""
    with engine.connect() as conn:
        apps = conn.execute(text("SELECT COUNT(*) FROM jobs")).scalar()
        mistakes = conn.execute(text("SELECT mistake FROM interview_experiences LIMIT 3")).fetchall()
    return f"Raw Data: {apps} apps sent. Recent mistakes: {[m[0] for m in mistakes]}."

# ==========================================
# HOUR 3: MCP CALENDAR SETUP
# ==========================================
def get_calendar_mcp_toolset():
    """Configures the connection to a remote Calendar MCP server."""
    if not mcp_available:
        return [] # Return empty if MCP is not installed to prevent crashes
    return MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=os.getenv("CALENDAR_MCP_URL", "http://mock-calendar-mcp-url"),
            headers={"Authorization": "Bearer YOUR_OAUTH_TOKEN"}
        )
    )

calendar_toolset = get_calendar_mcp_toolset()
root_tools = [log_job_application, audit_resume_against_jd, suggest_dsa_problem]
if calendar_toolset:
    root_tools.append(calendar_toolset)

# ==========================================
# HOUR 3: SEQUENTIAL WORKFLOW (Readiness Report)
# ==========================================
# Step 1: The Researcher Agent
progress_researcher = Agent(
    name="progress_researcher",
    model="gemini-2.5-flash",
    description="Gathers raw statistics from the database.",
    instruction="""
    You are a data researcher. Use the 'analyze_my_progress' tool to fetch the user's 
    job application count and recent interview mistakes. Output the raw data.
    """,
    tools=[analyze_my_progress],
    output_key="readiness_data" # Stores findings in shared state
)

# Step 2: The Presenter Agent
progress_presenter = Agent(
    name="progress_presenter",
    model="gemini-2.5-flash",
    description="Synthesizes raw data into a highly personalized readiness report.",
    instruction="""
    You are a career coach helping an MTech in Computer Science Engineering (CSE) student prepare for off-campus placement in the Delhi NCR region.
    Take the READINESS_DATA and format it into a highly motivating weekly report. 
    Remind them to stay focused on their goal of being job-ready by April 5, 2026, so they can start hitting their off-campus targets heavily on April 6.
    
    READINESS_DATA:
    {readiness_data}
    """
)

# Step 3: The Sequential Workflow
progress_workflow = SequentialAgent(
    name="progress_workflow",
    description="The main workflow for analyzing career progress.",
    sub_agents=[progress_researcher, progress_presenter]
)

# ==========================================
# THE ROOT AGENT (Command Center)
# ==========================================
root_agent = Agent(
    name="career_commander",
    model="gemini-2.5-flash",
    description="Your personal career and job readiness assistant.",
    instruction="""You are a Career Assistant. 
    - Use 'log_job_application' to track jobs. 
    - Use 'audit_resume_against_jd' for resume feedback.
    - Use 'suggest_dsa_problem' when a user mentions a coding mistake.
    - If MCP tools are available, use them to schedule calendar events for interviews.
    - When a user asks for a readiness or progress report, transfer control to the 'progress_workflow'.
    """,
    tools=root_tools,
    sub_agents=[progress_workflow]
)