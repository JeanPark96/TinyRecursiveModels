import open_clip
import torch

# def load_clip_models(model_name: str):
#     # models_list = clip.available_models()
#     models_list = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

#     for mn in models_list:
#         print(mn)
#         if mn != model_name:
#             continue
#         model, preprocess = clip.load(model_name)
#         model.cuda().eval()
#         input_resolution = model.visual.input_resolution
#         context_length = model.context_length
#         vocab_size = model.vocab_size

#         print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
#         print("Input resolution:", input_resolution)
#         print("Context length:", context_length)
#         print("Vocab size:", vocab_size)
#     return model, preprocess

def load_open_clip_models(model_name: str):
    # models_list, _, _ = open_clip.list_models()
    models_list = ['RN50', 'RN101', 'ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14']
    for mn in models_list:
        print(mn)
        if mn != model_name:
            continue
        model, _, preprocess = open_clip.create_model_and_transforms(mn, pretrained='openai')
        model.cuda().eval()
        # input_resolution = model.visual.input_resolution
        # context_length = model.context_length
        # vocab_size = model.token_embedding.num_embeddings

        print("Model parameters:", f"{sum([p.numel() for p in model.parameters()]):,}")
        # print("Input resolution:", input_resolution)
        # print("Context length:", context_length)
        # print("Vocab size:", vocab_size)
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')

if __name__ == "__main__":
    load_open_clip_models('ViT-B-16')