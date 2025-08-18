from dotenv import load_dotenv
from fastmcp import FastMCP
from time_mcp.models.NLinear import n_linear_forecast
import click
import uvicorn

forecast_mcp = FastMCP(name="forecast mcp")

classification_mcp = FastMCP(name="classification mcp")

anomaly_detection_mcp = FastMCP(name="anomaly detection mcp")

imputation_mcp = FastMCP(name="imputation mcp")
time_mcp = FastMCP(name="Timeseries MCP server")
time_mcp.mount(forecast_mcp, prefix="forecast")
time_mcp.mount(anomaly_detection_mcp, prefix="anomaly_detection")
time_mcp.mount(imputation_mcp, prefix="imputation")
time_mcp.mount(classification_mcp, prefix="classification")
forecast_mcp.tool(n_linear_forecast)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--host", type=str, default="127.0.0.1")
@click.option("--port", type=int, default="8021")
@click.option("--env_file", type=click.Path(exists=True), default=".env.example")
def http(host, port, env_file):
    load_dotenv(env_file)
    uvicorn.run(time_mcp.streamable_http_app(), host=host, port=port)


@cli.command()
@click.option("--env_file", type=click.Path(exists=True), default=".env.example")
def stdio(env_file):
    load_dotenv(env_file)
    time_mcp.run("stdio")
