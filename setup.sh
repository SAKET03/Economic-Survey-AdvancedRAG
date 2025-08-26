apt update && apt upgrade -y
apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/install.sh | sh

apt-get update
apt-get install -y libgl1
apt-get install -y libglib2.0-0