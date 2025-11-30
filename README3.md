sudo apt update && sudo apt install -y git python3 python3-venv python3-pip build-essential
cd /opt
sudo git clone https://example.com/ML-Web.git
sudo chown -R $USER:$USER ML-Web
cd ML-Web
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
make install
make eda
make train
make train-transformer
make evaluate
make test
make serve
make feedback-export
make history-report
docker compose up --build -d
