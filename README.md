## Getting started
I used "uv" to set up my virtual environment (Windows 11).
Either set up your own virtual environment or add the requirements to PATH.<br>

My steps:

In the terminal (I used PowerShell since I'm on Windows, syntax is slightly different for MACOS/bash), at the directory which will contain the virtual environment:

```powershell
uv venv --python 3.13
.\.venv\Scripts\activate
uv init
uv add -r .\requirements.txt
```

Next, you need to add an environment file alongside the .py files.
Create ".env" and add the following:

```
OPENAI_API_KEY=
```

And paste your personal key directly after the "=", with no other changes.

That should be it, from the dir with .venv and the code files, run the app with:

```powershell
streamlit run app.py
```

As of writing this README, I've sent 78 requests via my key during testing and
my usage is still $0.00, so in its current state, this agent is still very
cheap to run.