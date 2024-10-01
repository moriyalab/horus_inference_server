#!/bin/bash
mkdir -p ml_model

# embryo model
gdown --fuzzy -O ./ml_model/demo_model.pt 'https://docs.google.com/uc?export=download&id=1mnEUifj4NQXptB3rL48a8EarYtGrUqJx'

# coco train model
gdown --fuzzy -O ./ml_model/coco.pt 'https://docs.google.com/uc?export=download&id=1meIWpfRB0KYe7OcGC8j63dVCZGLKGOUb'