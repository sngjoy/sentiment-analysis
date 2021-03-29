from dataclasses import dataclass


@dataclass
class Model:
    embedding_path = None

    def build():
        model = Sequential(
            [
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.emb_dim,
                    input_length=self.max_length  # ,
                    # weights=[self.weight_matrix],
                    # embeddings_initializer=Constant(self.weight_matrix),
                    # trainable=True
                ),
                LSTM(
                    units=64,
                    dropout=0.0,  # units to drop for the linear trf of input
                    recurrent_dropout=0.0,  # units to drop for the linear trf of recurrent state
                ),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.summary()
        return model
