from train.cls.train_cls import train

if __name__ == "__main__":
    train(
        img_train= "G:\\UNet\\data\\ucf24_mini\\train",
        img_val= "G:\\UNet\\data\\ucf24_mini\\test",
        save_path=r"G:\UNet\result\train\cls\best_unet_n_lora.pth",
        model_type= "n",
        epochs= 10,
        batch_size= 12,
        pretrained=r"G:\UNet\result\train\cls\best_unet_n.pth",
        freeze_backbone=True,
        use_lora=True,
    )