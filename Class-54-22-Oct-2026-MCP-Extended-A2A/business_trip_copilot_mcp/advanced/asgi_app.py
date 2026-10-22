from servers.travel_server import server

# Deployable ASGI app for Streamable HTTP.
app = server.streamable_http_app(
    stateless_http=True,
    json_response=True,
)
