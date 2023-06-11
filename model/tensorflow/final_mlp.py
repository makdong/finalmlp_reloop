import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True, name="MLP"):
        super().__init__(name=name)

        self.dim_out = dim_out
        self.blocks = tf.keras.Sequential()
        for _ in range(num_hidden):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

            if batch_norm:
                self.blocks.add(tf.keras.layers.BatchNormalization())

            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.blocks.add(tf.keras.layers.Dense(dim_out))

    def call(self, inputs, training=False):
        out = self.blocks(inputs, training=training)

        return out


class FeatureSelection(tf.keras.Model):
    def __init__(self, dim_feature, dim_gate, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()

        self.gate_1 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_1",
        )
        self.gate_1_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

        self.gate_2 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_feature,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_2",
        )
        self.gate_2_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

    def call(self, embeddings):
        # embeddings is of shape (batch_size, dim_feature)
        gate_score_1 = self.gate_1(self.gate_1_bias)  # (bs, dim_feature)
        out_1 = 2.0 * tf.nn.sigmoid(gate_score_1) * embeddings  # (bs, dim_feature)

        gate_score_2 = self.gate_2(self.gate_2_bias)  # (bs, dim_feature)
        out_2 = 2.0 * tf.nn.sigmoid(gate_score_2) * embeddings  # (bs, dim_feature)

        return out_1, out_2  # (bs, dim_feature), (bs, dim_feature)


class Aggregation(tf.keras.Model):
    def __init__(self, dim_latent_1, dim_latent_2, num_heads=1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head_1 = dim_latent_1 // num_heads
        self.dim_head_2 = dim_latent_2 // num_heads

        self.w_1 = tf.keras.layers.Dense(1)
        self.w_2 = tf.keras.layers.Dense(1)
        self.w_12 = self.add_weight(
            shape=(num_heads, self.dim_head_1, self.dim_head_2, 1), initializer="glorot_normal", trainable=True
        )

    def call(self, latent_1, latent_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        first_order = self.w_1(latent_1) + self.w_2(latent_2)  # (bs, 1)

        latent_1 = tf.reshape(latent_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        latent_2 = tf.reshape(latent_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)
        second_order = tf.einsum("bhi,hijo,bhj->bho", latent_1, self.w_12, latent_2)  # (bs, num_heads, 1)
        second_order = tf.reduce_sum(second_order, 1)  # (bs, 1)

        out = first_order + second_order  # (bs, 1)

        return out


class FinalMLP(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=32,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        num_hidden_2=2,
        dim_hidden_2=64,
        num_heads=1,
        dropout=0.0,
        name="FinalMLP",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            name="embedding",
        )

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_feature=dim_input * dim_embedding,
            dim_gate=dim_input,
            dim_hidden=dim_hidden_fs,
            dropout=0.0,
        )

        # branch 1
        self.interaction_1 = MLP(
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
            name="MLP_1",
        )
        # branch 2
        self.interaction_2 = MLP(
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
            name="MLP_2",
        )

        # final aggregation layer
        self.aggregation = Aggregation(dim_latent_1=dim_hidden_1, dim_latent_2=dim_hidden_2, num_heads=num_heads)

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)  # (bs, num_emb, dim_emb)

        # (bs, num_emb * dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings)  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1, latent_2 = self.interaction_1(emb_1), self.interaction_2(emb_2)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = tf.nn.sigmoid(logits)  # (bs, 1)

        return outputs
