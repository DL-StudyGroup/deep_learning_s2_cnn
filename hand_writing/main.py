# coding: utf-8
import util
import cnn
import torch
import PIL.ImageOps

# cnn.train_net()

cnn_params = torch.load("./param/cnn_params.pkl")
new_cnn = cnn.CNN()
new_cnn.load_state_dict(cnn_params)
new_cnn.eval()

test_x = cnn.get_test_x()
# print("test_x:", test_x)
print("test_x.shape:", test_x.shape)
test_output = new_cnn(test_x)
# pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
pred_y = torch.max(test_output, 1)[1].numpy().squeeze()
print("pred_y:", pred_y)


def pred_image_num(filename):
    img_matrix = util.img2matrix(filename)
    # print("img_martrix:", img_matrix)

    output = new_cnn(img_matrix)
    pred_y = torch.max(output, 1)[1].numpy()
    print("output:", pred_y)


def pred_matrix(img_matrix):
    output = new_cnn(img_matrix)
    pred_y = torch.max(output, 1)[1].numpy()
    print(pred_y)


pred_image_num("./data/5.png")
pred_image_num("./data/1.png")

# my.cnn.test()

exit(0)
