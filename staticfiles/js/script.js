
async function sendData() {
    const response = await fetch('/api/data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: 'Hello from JavaScript!' })
    });
    const data = await response.json();
    document.getElementById('response').innerText = data.message;
}
    