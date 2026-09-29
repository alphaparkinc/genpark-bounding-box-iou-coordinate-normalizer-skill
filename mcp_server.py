import sys
import json
from client import BoundingBoxNormalizer

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-bounding-box-iou-coordinate-normalizer-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "calculate_iou",
                        "description": "Calculates Intersection over Union between two bounding boxes [ymin, xmin, ymax, xmax]",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "box_a": {"type": "array", "items": {"type": "number"}},
                                "box_b": {"type": "array", "items": {"type": "number"}}
                            },
                            "required": ["box_a", "box_b"]
                        }
                    },
                    {
                        "name": "apply_nms",
                        "description": "Applies Non-Maximum Suppression to filter redundant overlapping detections",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "detections": {"type": "array", "items": {"type": "object"}},
                                "iou_threshold": {"type": "number", "default": 0.5}
                            },
                            "required": ["detections"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "calculate_iou":
            iou = BoundingBoxNormalizer.compute_iou(args.get("box_a", []), args.get("box_b", []))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(iou)}]}}
        elif name == "apply_nms":
            filtered = BoundingBoxNormalizer.non_max_suppression(args.get("detections", []), args.get("iou_threshold", 0.5))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(filtered, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()
