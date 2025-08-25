import subprocess

def query_ollama(prompt):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "tinyllama"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # force UTF-8 decoding
            errors="replace"    # replace unrecognized characters 
        )

        stdout, stderr = process.communicate(prompt)

        if process.returncode != 0:
            print("Ollama error:", stderr)
            return "Model failed to respond."

        return stdout.strip()

    except Exception as e:
        return f"Error running Ollama: {e}"
