from iPhoneCore import build
from iPhoneCore import infer


image_path = r'data/dbwy0.bmp'
config_path = r'config.yml'


def test():
    model_path = r'final_model.pt'
    config, model1, model2 = build(config_path, model_path)
    infer(image_path, 1)
    print('Test have done!')


if __name__ == '__main__':
    test()

