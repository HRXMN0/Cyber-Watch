import socket
import ssl
import time
import urllib.parse
import requests
from datetime import datetime

# Port list to check
PORTS_TO_CHECK = [80, 443, 8080, 21, 22]

SECURITY_HEADERS = {
    'Strict-Transport-Security': 'HSTS',
    'Content-Security-Policy': 'CSP',
    'X-Frame-Options': 'X-Frame-Options',
    'X-XSS-Protection': 'X-XSS-Protection',
    'Referrer-Policy': 'Referrer-Policy',
    'Permissions-Policy': 'Permissions-Policy'
}

def validate_domain(url):
    try:
        if not url.startswith('http'):
            url = 'http://' + url
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.split(':')[0]
        if not domain:
            return False, None, "Invalid URL format."
        
        # DNS Resolution
        ip = socket.gethostbyname(domain)
        return True, domain, ip
    except socket.gaierror:
        return False, None, "Unable to resolve hostname."
    except Exception as e:
        return False, None, f"Invalid domain entered: {str(e)}"

def check_open_ports(ip):
    open_ports = []
    for port in PORTS_TO_CHECK:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
            sock.close()
        except:
            pass
    return open_ports

def get_ssl_info(domain):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((domain, 443), timeout=3) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                pass
    except Exception:
        return {"valid": False, "issuer": "Unknown", "expired": True, "error": "SSL Connection Failed"}
        
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=3) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                valid_to = cert.get('notAfter')
                issuer = dict(x[0] for x in cert.get('issuer', []))
                org = issuer.get('organizationName', 'Unknown')
                return {"valid": True, "issuer": org, "expired": False, "valid_to": valid_to}
    except ssl.SSLCertVerificationError as e:
        return {"valid": False, "issuer": "Unknown", "expired": "expired" in str(e).lower(), "error": str(e)}
    except Exception as e:
        return {"valid": False, "issuer": "Unknown", "expired": False, "error": str(e)}

def analyze_headers(url):
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        start_time = time.time()
        resp = requests.get(url, timeout=5, verify=False)
        latency = (time.time() - start_time) * 1000 # ms
        
        headers = resp.headers
        server = headers.get('Server', 'Unknown')
        framework = headers.get('X-Powered-By', 'Unknown')
        
        # Analyze security headers
        sec_headers = {}
        missing_count = 0
        for h in SECURITY_HEADERS.keys():
            if h in headers:
                sec_headers[h] = "Present"
            else:
                sec_headers[h] = "Missing"
                missing_count += 1
                
        return {
            "latency": latency,
            "server": server,
            "framework": framework,
            "security_headers": sec_headers,
            "missing_count": missing_count,
            "status_code": resp.status_code
        }
    except requests.exceptions.RequestException as e:
        return {
            "latency": 0,
            "server": "Unknown",
            "framework": "Unknown",
            "security_headers": {h: "Missing" for h in SECURITY_HEADERS},
            "missing_count": len(SECURITY_HEADERS),
            "status_code": 0,
            "error": str(e)
        }

def calculate_threat_score(ssl_info, ports, header_info, domain_rep=0):
    score = 0
    issues = []
    
    # Missing headers: +10 risk (cap at ~40)
    missing = header_info.get('missing_count', 0)
    if missing > 0:
        score += min(missing * 10, 40)
        issues.append(f"Missing {missing} critical security headers")
        
    # SSL
    if not ssl_info.get('valid'):
        score += 20
        issues.append("SSL configuration invalid or expired")
        
    # Open insecure ports
    for p in ports:
        if p in [21, 22, 8080]:
            score += 15
            issues.append(f"Open insecure port detected ({p})")
        elif p == 80:
            score += 5
            
    # Reputation
    if domain_rep > 0:
        score += 40
        issues.append("Domain flagged in reputation databases")
        
    if header_info.get('status_code') == 0:
        score += 20
        issues.append("Website unreachable or timing out")
        
    score = min(score, 100)
    
    # Category
    if score <= 20: level = "Safe"
    elif score <= 40: level = "Low Risk"
    elif score <= 60: level = "Moderate"
    elif score <= 80: level = "High Risk"
    else: level = "Critical"
    
    return score, level, issues

def run_scan(url):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    valid, domain, ip = validate_domain(url)
    if not valid:
        return {"error": ip} # ip holds the error message here
        
    ports = check_open_ports(ip)
    ssl_info = get_ssl_info(domain)
    header_info = analyze_headers(url)
    
    score, level, issues = calculate_threat_score(ssl_info, ports, header_info)
    
    # Fake radar chart data for exposure
    radar = {
        "network": min(len(ports)*25, 100),
        "application": (header_info.get('missing_count',0) / max(1, len(SECURITY_HEADERS))) * 100,
        "reputation": 0, 
        "malware": 0,    
        "traffic": min(header_info.get('latency', 0) / 10, 100)
    }
    
    return {
        "domain": domain,
        "ip": ip,
        "server": header_info.get('server'),
        "framework": header_info.get('framework'),
        "ssl_valid": ssl_info.get('valid'),
        "ssl_detail": ssl_info,
        "security_headers": header_info.get('security_headers'),
        "latency": header_info.get('latency'),
        "open_ports": ports,
        "threat_score": score,
        "threat_level": level,
        "issues": issues,
        "radar": radar
    }
