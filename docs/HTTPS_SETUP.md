# HTTPS Setup Guide

This guide explains how to set up HTTPS for the Multimodal Emotion Analysis System.

## Why HTTPS?

Modern browsers require HTTPS (or localhost) to access the microphone. This is a security feature to protect users' privacy.

## Quick Start (Development)

For development, you can use Flask's built-in ad-hoc SSL certificate:

```bash
python app.py --ssl-adhoc
```

This will:
- Generate a self-signed certificate automatically
- Start the server on `https://localhost:5001`
- Show a browser security warning (click "Advanced" → "Proceed" to continue)

**Note:** The browser will show a security warning because it's self-signed. This is normal for development.

## Production Setup

For production, you need proper SSL certificates. Here are your options:

### Option 1: Use Let's Encrypt (Free)

1. Install certbot:
```bash
# Ubuntu/Debian
sudo apt-get install certbot

# macOS
brew install certbot
```

2. Generate certificates:
```bash
sudo certbot certonly --standalone -d yourdomain.com
```

3. Run the server:
```bash
python app.py --ssl-cert /etc/letsencrypt/live/yourdomain.com/fullchain.pem --ssl-key /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

### Option 2: Use Your Own Certificates

If you have your own SSL certificates:

```bash
python app.py --ssl-cert /path/to/certificate.pem --ssl-key /path/to/private-key.pem
```

### Option 3: Use a Reverse Proxy (Recommended for Production)

Use nginx or Apache as a reverse proxy with SSL termination:

**nginx example:**
```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /path/to/certificate.pem;
    ssl_certificate_key /path/to/private-key.pem;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Then run Flask on HTTP (no SSL):
```bash
python app.py --host 127.0.0.1
```

## Command Line Options

```bash
python app.py [OPTIONS]

Options:
  --port PORT          Port to run the server on (default: 5001)
  --host HOST          Host to bind to (default: 0.0.0.0)
  --ssl-cert PATH      Path to SSL certificate file
  --ssl-key PATH       Path to SSL private key file
  --ssl-adhoc          Use ad-hoc SSL certificate (for development)
```

## Examples

### Development with HTTPS:
```bash
python app.py --ssl-adhoc
# Access at: https://localhost:5001
```

### Production with custom certificates:
```bash
python app.py --ssl-cert /etc/ssl/certs/server.crt --ssl-key /etc/ssl/private/server.key --port 443
# Access at: https://yourdomain.com
```

### HTTP only (localhost only, no microphone):
```bash
python app.py
# Access at: http://localhost:5001
# Note: Microphone won't work without HTTPS (except on localhost)
```

## Troubleshooting

### Browser shows "Not Secure" warning
- This is normal for self-signed certificates (--ssl-adhoc)
- Click "Advanced" → "Proceed to localhost" to continue
- For production, use proper certificates from a CA

### Microphone still not working
- Make sure you're accessing via `https://` (not `http://`)
- Check browser console for errors
- Verify microphone permissions in browser settings
- Try a different browser

### Certificate errors
- Make sure certificate and key paths are correct
- Check file permissions (certificate should be readable)
- Verify certificate hasn't expired

## Why Permission Prompts?

**This is normal browser behavior!** Browsers always ask for microphone permission:
- **First time**: Browser will always prompt
- **After clearing data**: Permission resets, browser will ask again
- **Different domain**: Each domain needs its own permission

The permission prompt is a **security feature**, not a bug. It protects users' privacy by requiring explicit consent before accessing the microphone.

## Localhost Exception

Browsers allow microphone access on `localhost` and `127.0.0.1` even without HTTPS. So if you're developing locally, you can use:

```bash
python app.py
# Access at: http://localhost:5001
```

Microphone will work without HTTPS on localhost!

