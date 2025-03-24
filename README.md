# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
 Import required libraries, load the dataset, and define training & testing datasets.
</br>


### STEP 2:
</br>
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model using the training dataset with forward and backward propagation.
</br>

### STEP 4:
Evaluate the model on the testing dataset to measure accuracy and performance.
<br/>

### STEP 5:
 Make predictions on new data using the trained model.
 <br/>

## PROGRAM

```
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

```


# Modify the final fully connected layer to match the dataset classes
```python
num_classes = len(train_dataset.classes)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features,1)
```
# Include the Loss function and optimizer
```python
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)
```



# Train the model
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float() )
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:MADESWARAN M")
    print("Register Number:212223040106")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```




## OUTPUT
### Training Loss
</br>
</br>
</br>

![Screenshot 2025-03-24 154114](https://github.com/user-attachments/assets/532ef4e2-b816-4ffd-b8cb-c9efd2c71297)


###  Validation Loss Vs Iteration Plot
![Screenshot 2025-03-24 154138](https://github.com/user-attachments/assets/0287a5af-642b-427a-9ffb-4b12aa6a49a8)



### Confusion Matrix
![Screenshot 2025-03-24 154239](https://github.com/user-attachments/assets/8a8c5139-725f-4a30-b462-947b00ade1b3)


</br>
</br>
</br>

### Classification Report

![Screenshot 2025-03-24 154258](https://github.com/user-attachments/assets/12292ae9-1f29-4b00-9518-beb957a2974f)
</br>
</br>
</br>


### New Sample Prediction
```
predict_image(model, image_index=65, dataset=test_dataset)
```
![image-4](https://github.com/user-attachments/assets/6898ac62-4dc3-4829-97d7-312c7b045c2f)


```

predict_image(model, image_index=95, dataset=test_dataset)
```
![image-5](https://github.com/user-attachments/assets/1ab7aca9-3da3-44b7-a406-0791adb0e6f6)

</br>
</br>
</br>

## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
</br>
</br>
</br>
