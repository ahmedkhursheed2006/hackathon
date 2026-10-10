document.addEventListener('DOMContentLoaded', () => {
    const wsIndicator = document.getElementById('ws-indicator');
    const wsText = document.getElementById('ws-text');
    const statusCard = document.getElementById('status-card');
    const statusHeading = document.getElementById('status-heading');
    const statusDesc = document.getElementById('status-desc');
    const activityLog = document.getElementById('activity-log');

    let ws = null;
    let reconnectInterval = null;
    const WS_URL = 'ws://127.0.0.1:8080/ws';

    function addLogEntry(message, type = 'normal') {
        const li = document.createElement('li');
        const time = new Date().toLocaleTimeString();
        let colorClass = '';
        if (type === 'safe') colorClass = 'log-safe';
        if (type === 'threat') colorClass = 'log-threat';
        
        li.innerHTML = `<span class="log-time">[${time}]</span> <span class="${colorClass}">${message}</span>`;
        activityLog.insertBefore(li, activityLog.firstChild);
        
        // Keep only last 20 logs
        if (activityLog.children.length > 20) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }

    function updateStatus(status) {
        if (status === 'THREAT_DETECTED') {
            statusCard.className = 'status-card threat';
            statusHeading.textContent = 'Threat Detected';
            statusDesc.textContent = 'Shoulder surfer detected! Privacy shield activated.';
            addLogEntry('Threat detected - Shield active', 'threat');
        } else if (status === 'SAFE') {
            statusCard.className = 'status-card safe';
            statusHeading.textContent = 'System Safe';
            statusDesc.textContent = 'No shoulder surfers detected.';
            addLogEntry('Environment clear - System safe', 'safe');
        }
    }

    function connect() {
        if (ws) {
            ws.close();
        }

        wsIndicator.className = 'indicator offline';
        wsText.textContent = 'Connecting...';

        try {
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                wsIndicator.className = 'indicator online';
                wsText.textContent = 'Connected';
                addLogEntry('WebSocket connected to Medusa Guardian');
                
                // Keep it neutral/safe upon initial connect until status event
                statusCard.className = 'status-card safe';
                statusHeading.textContent = 'Connected';
                statusDesc.textContent = 'Awaiting status from Guardian Core...';
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data && data.status) {
                        updateStatus(data.status);
                    }
                } catch (e) {
                    console.error('Error parsing WS message:', e);
                }
            };

            ws.onclose = () => {
                wsIndicator.className = 'indicator error';
                wsText.textContent = 'Disconnected';
                statusCard.className = 'status-card offline';
                statusHeading.textContent = 'Offline';
                statusDesc.textContent = 'Waiting for connection to Guardian Core...';
                
                if (!reconnectInterval) {
                    addLogEntry('WebSocket disconnected, reconnecting...', 'normal');
                    reconnectInterval = setInterval(connect, 3000);
                }
            };

            ws.onerror = (err) => {
                console.error('WebSocket Error:', err);
                ws.close();
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            if (!reconnectInterval) {
                reconnectInterval = setInterval(connect, 3000);
            }
        }
    }

    // Initialize
    connect();
});
