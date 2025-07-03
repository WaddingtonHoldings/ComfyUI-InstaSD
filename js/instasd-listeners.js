import { app } from "../../scripts/app.js";

// Register the extension
app.registerExtension({
    name: "InstaSD.Listeners",
    
    async setup() {
        // Add message listeners with origin checking
        window.addEventListener("message", async (event) => {
            // Allow any localhost origin including subdomains and any instasd.com subdomain
            if (!event.origin.match(/^https?:\/\/(.*\.)?localhost(:[0-9]+)?$/) && 
                !event.origin.match(/^https?:\/\/(.*\.)?instasd\.com$/) &&
                !event.origin.match(/^https?:\/\/(.*\.)?instasddev\.com$/)) {
                console.log("Rejected message from unauthorized origin:", event.origin);
                return;
            }

            if (event.data.type === "health_check") {
                event.source.postMessage({ type: "health_response", status: "ok" }, event.origin);
            } else if (event.data.type === "instasd_receive_graph") {
                try {
                    sessionStorage.clear();
                    const graphData = JSON.parse(event.data.graph);
                    const workflowName = event.data.workflowName;
                    app.loadGraphData(graphData, true, true, workflowName);
                    event.source.postMessage({ 
                        type: "workflow_saved", 
                        status: "success" 
                    }, event.origin);
                } catch (error) {
                    console.error('Error loading workflow:', error);
                    event.source.postMessage({ 
                        type: "workflow_saved", 
                        error: "Failed to parse or load workflow" 
                    }, event.origin);
                }
            } else if (event.data.type === "get_enabled_node_lists") {
                try {
                    // Fetch both customnode list and object_info in parallel
                    const [nodeResponse, objectInfoResponse, extensions] = await Promise.all([
                        fetch('/api/customnode/getlist?mode=cache&skip_update=true'),
                        fetch('/api/object_info'),
                        fetch('/api/extensions')
                    ]);
                    
                    const nodeData = await nodeResponse.json();
                    const objectInfo = await objectInfoResponse.json();
                    const comfyExtensions = await extensions.json();
                    
                    // Extract IDs of nodes with state "enabled"
                    const enabledNodeIds = Object.keys(nodeData.node_packs)
                        .filter(nodeId => nodeData.node_packs[nodeId].state === "enabled")
                        .map(nodeId => nodeData.node_packs[nodeId].id);

                    event.source.postMessage({ 
                        type: "node_lists_response", 
                        nodes: enabledNodeIds,
                        object_info: JSON.stringify(objectInfo),
                        comfy_extensions: JSON.stringify(comfyExtensions)
                    }, event.origin);
                } catch (error) {
                    console.error('Error fetching node lists or object info:', error);
                    event.source.postMessage({ 
                        type: "node_lists_response", 
                        error: "Failed to fetch node lists or object info" 
                    }, event.origin);
                }
            }
        });

        window.addEventListener("message", async (event) => {
            // Allow any localhost origin including subdomains and any instasd.com subdomain
            if (!event.origin.match(/^https?:\/\/(.*\.)?localhost(:[0-9]+)?$/) && 
                !event.origin.match(/^https?:\/\/(.*\.)?instasd\.com$/) &&
                !event.origin.match(/^https?:\/\/(.*\.)?instasddev\.com$/)) {
                console.log("Rejected message from unauthorized origin:", event.origin);
                return;
            }

            if (event.data.type === "instasd_get_current_graph") {
                try {
                    // Serialize the graph to JSON
                    const p = await app.graphToPrompt();
                    const graph_json = JSON.stringify(p["workflow"], null, 2);
                    const api_json = JSON.stringify(p["output"], null, 2);
                    event.source.postMessage({ 
                        type: "graph_response", 
                        graph_json: graph_json,
                        api_json: api_json
                    }, event.origin);
                } catch (error) {
                    console.error('Error processing graph:', error);
                    event.source.postMessage({ 
                        type: "graph_response", 
                        error: "Failed to process graph" 
                    }, event.origin);
                }
            }
        });
    }
}); 