from dotenv import load_dotenv

# Live tests (-m live) read API keys from .env; regular tests never touch the network.
load_dotenv()
