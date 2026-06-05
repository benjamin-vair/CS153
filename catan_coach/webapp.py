"""Interactive board UI for the opening advisor.

A tiny local web app (Python stdlib only -- no extra installs) that lets you
build a Catan board by clicking, place pieces, and get the best-next-settlement
recommendation from the trained value net.

    cd catan_coach
    ./.venv/bin/python webapp.py        # then open http://127.0.0.1:8000

All board geometry is computed here (server side) from Catanatron's own node
coordinates and handed to the browser, so the drawing always matches the
engine's node IDs. The /api/advise endpoint reuses advise.py end to end.
"""

import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from catanatron import Game, Color, RandomPlayer

import advise as A
from model import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 56  # hex radius in px
ANGLES = {  # NodeRef -> corner angle (deg), pointy-top; verified vs engine geometry
    "NORTH": 90, "NORTHEAST": 30, "SOUTHEAST": 330,
    "SOUTH": 270, "SOUTHWEST": 210, "NORTHWEST": 150,
}

# Load the value net once at startup.
MODEL, MEAN, STD = load_checkpoint(os.path.join(HERE, "valuenet_v2.pt"))


def _ref(k):
    return str(k).split(".")[-1]


def compute_layout():
    """Pixel geometry for hexes, nodes and ports (topology is fixed)."""
    g = Game([RandomPlayer(c) for c in A.SEAT_COLORS], seed=0)
    bmap = g.state.board.map

    def center(coord):
        x, _y, z = coord
        return SIZE * math.sqrt(3) * (x + z / 2), SIZE * 1.5 * z

    node_xy = {}
    hexes = []
    for idx, coord in enumerate(A.LAND_COORDS):  # 19 land hexes, standard order
        cx, cy = center(coord)
        tile = bmap.tiles[coord]
        corners = []
        for ang in (30, 90, 150, 210, 270, 330):
            corners.append([cx + SIZE * math.cos(math.radians(ang)),
                            cy - SIZE * math.sin(math.radians(ang))])
        for k, nid in tile.nodes.items():
            a = math.radians(ANGLES[_ref(k)])
            node_xy[nid] = [cx + SIZE * math.cos(a), cy - SIZE * math.sin(a)]
        hexes.append({"index": idx, "cx": cx, "cy": cy, "corners": corners})

    # ports: mark the land nodes that touch a port
    ports = []
    for resource, nodes in bmap.port_nodes.items():
        label = "3:1" if resource is None else resource[:2]
        for nid in nodes:
            if nid in node_xy:
                ports.append({"node": nid, "label": label})

    xs = [p[0] for p in node_xy.values()]
    ys = [p[1] for p in node_xy.values()]
    pad = SIZE
    minx, miny = min(xs) - pad, min(ys) - pad
    w, h = (max(xs) - minx) + pad, (max(ys) - miny) + pad
    return {
        "size": SIZE,
        "viewBox": [minx, miny, w, h],
        "hexes": hexes,
        "nodes": {str(k): v for k, v in node_xy.items()},
        "ports": ports,
        "defaultTiles": A.EXAMPLE["tiles"],  # prefill a full board to edit
    }


LAYOUT = compute_layout()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "board_ui.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/layout":
            self._send(200, json.dumps(LAYOUT))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/advise":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            spec = json.loads(self.rfile.read(n) or b"{}")
            game = A.build_game(spec)
            recs = A.advise(game, spec["advise_for"], MODEL, MEAN, STD, top_k=999)
            self._send(200, json.dumps({"recs": recs}))
        except Exception as e:  # surface engine/validation errors to the UI
            self._send(400, json.dumps({"error": str(e)}))


def main(port=8000):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Catan Coach UI  ->  http://127.0.0.1:{port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
