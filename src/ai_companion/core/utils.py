import re
from urllib.parse import urlparse, urlunparse, quote, unquote

def sanitize_string(input_string: str) -> str:
    """Remove non-printable characters from a string, including carriage returns and newlines."""
    if not isinstance(input_string, str):
        return str(input_string)
    # First remove all non-printable characters including \r, \n, \t
    cleaned = ''.join(char for char in input_string if char.isprintable() and char not in '\r\n\t')
    # Then remove any remaining control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    # Remove any trailing/leading whitespace
    cleaned = cleaned.strip()
    # Remove any multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def clean_url(url: str) -> str:
    """Clean the URL by removing unwanted characters and properly encoding it."""
    if not isinstance(url, str):
        url = str(url)
    
    try:
        print(f"Original URL: {repr(url)}")
        
        # First sanitize the string
        cleaned_url = sanitize_string(url)
        print(f"After sanitize_string: {repr(cleaned_url)}")
        
        # First unquote to handle any existing encoding
        cleaned_url = unquote(cleaned_url)
        print(f"After unquote: {repr(cleaned_url)}")
        
        # Split the URL into components
        parsed = urlparse(cleaned_url)
        
        # Encode the path and query components
        encoded_path = quote(parsed.path, safe='/:')
        encoded_query = quote(parsed.query, safe='=&')
        
        # Reconstruct the URL with encoded components
        cleaned_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            encoded_query,
            parsed.fragment
        ))
        
        print(f"After URL parsing: {repr(cleaned_url)}")
        
        # Final validation to ensure URL is compatible with httpx
        try:
            from httpx import URL
            URL(cleaned_url)  # This will raise InvalidURL if there are still issues
        except Exception as e:
            print(f"httpx URL validation failed: {e}")
            # If httpx validation fails, try one more aggressive cleaning
            cleaned_url = re.sub(r'[^\x20-\x7E]', '', cleaned_url)
            cleaned_url = quote(cleaned_url, safe=':/?=&')
            # Try one more time with httpx
            try:
                URL(cleaned_url)
            except Exception as e:
                print(f"Second httpx validation failed: {e}")
                # If still failing, try one last time with minimal cleaning
                cleaned_url = re.sub(r'[^\x20-\x7E]', '', url)
                cleaned_url = quote(cleaned_url, safe=':/?=&')
        
        print(f"Final cleaned URL: {repr(cleaned_url)}")
        return cleaned_url
    except Exception as e:
        print(f"Error cleaning URL: {e}")
        print(f"Problematic URL: {repr(url)}")
        # Try one last time with minimal cleaning
        try:
            cleaned_url = re.sub(r'[^\x20-\x7E]', '', url)
            cleaned_url = quote(cleaned_url, safe=':/?=&')
            return cleaned_url
        except:
            return url  # Return original URL if all cleaning attempts fail

class URLValidator:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:'  # Start of group for domain/IP/localhost
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'  # domain
            r'|localhost'  # or localhost
            r'|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # or IPv4
        r')'  # End of group for domain/IP/localhost
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$',  # path
        re.IGNORECASE
    )

    @classmethod
    def is_valid(cls, url: str) -> bool:
        try:
            # Always try to clean the URL first
            cleaned_url = clean_url(url)
            # Then validate it
            return bool(cls.regex.match(cleaned_url))
        except Exception as e:
            print(f"URL validation error: {e}")
            return False 