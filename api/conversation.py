from flask import session

MAX_HISTORY = 10

def get_history(user_id):
    """
    Get the conversation history for the user.
    Returns a list of (user_message, bot_message) tuples.
    """
    return session.get(f"history_{user_id}", [])

def update_history(user_id, user_msg, bot_msg):
    """
    Update the conversation history for the user.
    Keeps only the most recent MAX_HISTORY exchanges.
    """
    history = session.get(f"history_{user_id}", [])
    history.append((user_msg, bot_msg))
    # Keep only the last MAX_HISTORY turns
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session[f"history_{user_id}"] = history

def format_history(history):
    """
    Format the conversation history so the user is always the question-asker
    and the bot is always the answerer.
    """
    prompt_lines = []
    for user_msg, bot_msg in history:
        prompt_lines.append(f"User: {user_msg}")
        prompt_lines.append(f"Bot: {bot_msg}")
    return "\n".join(prompt_lines)