from train import TrainerConfig, Trainer

def main_loop(trainer):
    print("--文字を入力してください--")
    sent = input().strip()
    print("--生成中--")
    sent = trainer.generate(sent)
    print(sent)

if __name__ == '__main__':
    config  = TrainerConfig.from_json("../config/train_config.json")
    trainer = Trainer(config)

    while True:
        main_loop(trainer)