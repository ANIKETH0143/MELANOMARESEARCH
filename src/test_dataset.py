# from dataset import get_loaders

# train_loader, val_loader, test_loader = get_loaders(batch_size=4)

# images, labels = next(iter(train_loader))

# print("Images:", images.shape)
# print("Labels:", labels.shape)
# print("Labels:", labels)




from dataset import get_loaders


train_loader, val_loader, test_loader = get_loaders(
    dataset_name="PH2",
    batch_size=4
)


images, labels = next(iter(train_loader))


print("Images:", images.shape)
print("Labels:", labels.shape)
print("Labels:", labels)