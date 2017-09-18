import elice_utils
import matplotlib.pyplot as plt

# Simple helper function to show an image and its landmarks

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    
    return
    
# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')
    plt.axis('off')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    
    return