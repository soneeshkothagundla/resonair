# Vercel Deployment Script 

Run this script to start the backend Streamlit server and Cloudflared tunnel securely. 
The Vercel link `https://resonair.vercel.app` will securely proxy to the Cloudflared tunnel.

```powershell
# Start Streamlit App in background
Start-Process powershell -ArgumentList "-NoExit -Command `"cd 'D:\STATA\ProjectWork\Chest Diseases Using Different Medical Imaging and Cough Sounds\Chest Diseases Dataset\Chest Diseases Dataset'; streamlit run app.py --server.port 8501`""

# Wait briefly
Start-Sleep -Seconds 5

# Start Cloudflare Tunnel pointing to Streamlit
cloudflared tunnel --url http://127.0.0.1:8501
```
