import sqlite3
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Sales-Database")

DB_PATH = "data/sales.db"

def init_db():
    # Simple setup for your example
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS sales (id INTEGER, item TEXT, amount REAL, date TEXT)")
    # Sample data
    conn.execute("INSERT INTO sales VALUES (1, 'Widget A', 500, '2024-10-15')")
    conn.commit()
    conn.close()

@mcp.tool()
def query_sales_db(sql_query: str) -> str:
    """Execute SQL queries on the sales database to get revenue and report data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return str(rows)
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    mcp.run()