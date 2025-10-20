
BLUE = "34"
PURPLE = "35"

def bold(text: str, color: str = BLUE) -> str:
    """Devuelve el texto formateado en negrita y color para la consola."""
    return f"\033[1m\033[{color}m{text}\033[0m"
