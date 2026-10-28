# Ollama for Reddragon AI in production

Tried to serve an LLM model a Windows PC powered by an AMD Radeon RX 7900 GRE.

original goal was to serve MiniCPM inside a custom FastAPI/PyTorch Docker container on Windows 11 using AMD GPU (ROCm) acceleration via WSL2.

My journey
1. Turn on SMV on bios, and my PC failed to POST twice, only worked after a hard reboot
2. download docker and set up
3. tried to spin up a docker container with miniCPM
4. hard crashed many times
5. pivoted to ollama to mat least serve some model


Key issues

- Whenever Docker attempted to push the massive model weights into the GPU, Windows popped up an AMD Bug Report Tool warning about a driver timeout, followed immediately by fatal errors like [CreateContext] fail c000000d.
- PC occasionally turned itself off completely right as the Docker container initialized the model.


For reference, the ollama only uses the docker-compose.yaml, i uploaded the failed fastAPI test as well.
