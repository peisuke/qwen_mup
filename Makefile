build:
	docker build -t mup .

run:
	docker run -it --gpus all -v "$$(pwd)/workspace:/workspace/tmp" mup /bin/bash
