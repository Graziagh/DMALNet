import numpy as np


def run_grid_attention_example(img_path="visualize/test_data/example.jpg",
                               save_path="test/",
                               attention_mask=None, quality=300):
    if not attention_mask:
        attention_mask = np.random.randn(14, 14)

    visulize_grid_attention_v2(img_path=img_path,
                               save_path=save_path,
                               attention_mask=attention_mask,
                               save_image=True,
                               save_original_image=False,
                               quality=quality)
