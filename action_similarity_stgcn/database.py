from action_similarity_stgcn.utils import exist_embeddings, parse_action_label, load_embeddings

class ActionDatabase():
    def __init__(
        self,
        database_path: str,
        label_path: str = None,
        target_actions=None,
        key = 'default'):

        assert exist_embeddings(embeddings_dir=database_path, key=key), \
                f"The embeddings(key = {key}) not exist. "\
                f"You should run the bin.preprocess before loading the action db"
        
        self.db = load_embeddings(
            key=key,
            embeddings_dir=database_path,
            target_actions=target_actions,
        )
        self.actions = parse_action_label(label_path)