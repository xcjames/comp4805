#step1: git clone https://github.com/xcjames/comp4805.git
#step2: run the following python command

Experiments commands:
Hyperparameter tuning:

python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer_weight 0.05;
python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer_weight 0.1;
python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer_weight 0.2;
python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer_weight 0.3;
python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer_weight 0.5;

python main.py --data yelp --add_noise_to_emb False --cl_crossLayer False --beta 0.02;
python main.py --data yelp --add_noise_to_emb False --cl_crossLayer False --beta 0.05;
python main.py --data yelp --add_noise_to_emb False --cl_crossLayer False --beta 0.1;
python main.py --data yelp --add_noise_to_emb False --cl_crossLayer False --beta 0.2;

python main.py --data yelp --denoise False --cl_crossLayer False --eps 0.01;
python main.py --data yelp --denoise False --cl_crossLayer False --eps 0.025;
python main.py --data yelp --denoise False --cl_crossLayer False --eps 0.05;
python main.py --data yelp --denoise False --cl_crossLayer False --eps 0.1;


On different datasets:

python main.py --data yelp --add_noise_to_emb False --denoise False --cl_crossLayer False;
python main.py --data yelp --eps 0.025 --beta 0.05 --cl_crossLayer_weight 0.1;

python main.py --data gowalla --add_noise_to_emb False --denoise False --cl_crossLayer False;
python main.py --data gowalla --eps 0.025 --beta 0.05 --cl_crossLayer_weight 0.1;
python main.py --data gowalla --eps 0.025 --beta 0.1 --cl_crossLayer_weight 0.2;

python main.py --data tmall --add_noise_to_emb False --denoise False --cl_crossLayer False --epoch 30;
python main.py --data tmall --eps 0.025 --beta 0.05 --cl_crossLayer_weight 0.1 --epoch 30;
python main.py --data tmall --eps 0.025 --beta 0.1 --cl_crossLayer_weight 0.2 --epoch 30;
