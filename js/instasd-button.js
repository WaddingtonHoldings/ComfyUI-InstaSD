import { app } from "../../scripts/app.js";

// Register the extension
app.registerExtension({
    name: "InstaSD.Button",
    
    // Add a badge to the about page
    aboutPageBadges: [
        {
            label: "InstaSD",
            url: 'https://github.com/yourusername/ComfyUI-InstaSD',
            icon: 'pi pi-camera' // You can choose an appropriate icon from PrimeIcons
        }
    ],
    
    // Add a command that can be triggered via keyboard shortcuts
    commands: [
        {
            id: "InstaSD.ToggleInterface",
            label: "Toggle InstaSD Interface",
            icon: "pi pi-camera",
            function: () => {
                // This function will be called when the command is triggered
                // You can implement your toggle functionality here
                console.log("InstaSD interface toggle triggered");
                // Example: if you have a dialog, you could toggle it here
                // if (instaSDDialog) instaSDDialog.toggleVisibility();
            },
        }
    ],
    
    async setup() {
        // Add message listeners with origin checking
        window.addEventListener("message", async (event) => {
            // Allow any localhost origin including subdomains
            if (!event.origin.match(/^https?:\/\/(.*\.)?localhost(:[0-9]+)?$/)) {
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
                    const response = await fetch('/api/customnode/getlist?mode=cache&skip_update=true');
                    const data = await response.json();
                    
                    // Extract IDs of nodes with state "enabled"
                    const enabledNodeIds = Object.keys(data.node_packs)
                        .filter(nodeId => data.node_packs[nodeId].state === "enabled")
                        .map(nodeId => data.node_packs[nodeId].id);

                    event.source.postMessage({ 
                        type: "node_lists_response", 
                        nodes: enabledNodeIds 
                    }, event.origin);
                } catch (error) {
                    console.error('Error fetching node lists:', error);
                    event.source.postMessage({ 
                        type: "node_lists_response", 
                        error: "Failed to fetch node lists" 
                    }, event.origin);
                }
            }
        });

        window.addEventListener("message", async (event) => {
            // Allow any localhost origin including subdomains
            if (!event.origin.match(/^https?:\/\/(.*\.)?localhost(:[0-9]+)?$/)) {
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

        try {
            // Try to use the new style buttons (for newer ComfyUI versions)
            const ButtonGroup = await import("../../scripts/ui/components/buttonGroup.js");
            const Button = await import("../../scripts/ui/components/button.js");
            
            // Create a button group with a main button and potentially additional buttons
            let instaSDGroup = new ButtonGroup.ComfyButtonGroup(
                new Button.ComfyButton({
                    icon: "camera", // Use an appropriate icon
                    action: () => {
                        // Main button action
                        console.log("InstaSD button clicked");
                        // Add your action here, e.g., open a dialog
                    },
                    tooltip: "InstaSD",
                    content: "InstaSD", // Text on the button
                    classList: "comfyui-button comfyui-menu-mobile-collapse primary"
                }).element
                // You can add more buttons to the group if needed
            );
            
            // Add the button group to the menu
            // This places it before the settings group
            app.menu?.settingsGroup.element.before(instaSDGroup.element);
        }
        catch(exception) {
            console.log('Using fallback button for older ComfyUI versions');
            
            // Fallback for older ComfyUI versions - add a traditional button
            const menu = document.querySelector(".comfy-menu");
            
            // Add a separator
            const separator = document.createElement("hr");
            separator.style.margin = "20px 0";
            separator.style.width = "100%";
            menu.append(separator);
            
            // Create and add the button
            const instaSDButton = document.createElement("button");
            instaSDButton.textContent = "InstaSD";
            instaSDButton.onclick = () => {
                console.log("InstaSD button clicked");
                // Add your action here
            };
            
            // Optional: Style the button to make it stand out
            instaSDButton.style.background = "linear-gradient(90deg, #FF6B6B 0%, #FFE66D 100%)";
            instaSDButton.style.color = "black";
            
            menu.append(instaSDButton);
        }
    }
}); 