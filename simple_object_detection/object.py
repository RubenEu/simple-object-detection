class Object:
    def __init__(self, index, bounding_box, score, label, **kwargs):
        """

        :param index: identificador único del objeto.
        :param bounding_box: caja delimitadora formada por 2 puntos, la esquina superior izquierda y la esquina inferior
                             derecha. Tipo: ndarray 4 posiciones de tipo uint32.
        :param score: puntuación del objeto (int/float).
        :param label: etiqueta (string) de la clase a la que pertenece el objeto.
        :param kwargs: otras propiedades.
        """
        self.index = index
        self.bounding_box = bounding_box
        self.score = score
        self.label = label
        self.other_properties = kwargs

    def get_centroid(self):
        """
        Calcula el centroide del objeto.

        :return: tupla de la posición en píxels (x, y).
        """
        (top_left_x, top_left_y) = (self.bounding_box[0], self.bounding_box[2])
        (bottom_right_x, bottom_right_y) = (self.bounding_box[1], self.bounding_box[3])
        center_x = top_left_x + 1/2 * (bottom_right_x - top_left_x)
        center_y = top_left_y + 1/2 * (bottom_right_y - top_left_y)
        return int(center_x), int(center_y)

    def get_bounding_box_width(self):
        """
        Devuelve el ancho de la caja delimitadora.

        :return: longitud de la caja en píxels.
        """
        (top_left_x, top_left_y) = (self.bounding_box[0], self.bounding_box[2])
        (bottom_right_x, bottom_right_y) = (self.bounding_box[1], self.bounding_box[3])
        return int(bottom_right_x - top_left_x)

    def get_bounding_box_height(self):
        """
        Devuelve el alto de la caja delimitadora.

        :return: alto de la caja en píxels.
        """
        (top_left_x, top_left_y) = (self.bounding_box[0], self.bounding_box[2])
        (bottom_right_x, bottom_right_y) = (self.bounding_box[1], self.bounding_box[3])
        return int(bottom_right_y - top_left_y)

    def is_labeled(self, class_name):
        """
        Comprueba si el objeto está etiquetado como de la clase 'class_name'.

        :param class_name: Nombre de la clase que se busca comprobar si pertenece a ella.
        :return: si pertenece o no a 'class_name'.
        """
        return class_name.lower() == self.label.lower()

    def is_greater_scored_than(self, score):
        """
        Comprueba si la puntuación del objeto es mayor que el parámetro score.

        :param score: puntuación con la que se quiere comparar.
        :return: bool indicando si se cumple la propiedad.
        """
        return self.score > score

    def is_greatereq_scored_than(self, score):
        """
        Comprueba si la puntuación del objeto es mayor estricto que el parámetro score.

        :param score: puntuación con la que se quiere comparar.
        :return: bool indicando si se cumple la propiedad.
        """
        return self.score >= score

    def __str__(self):
        return 'ObjectDetected<id={}, class={}, score={}>'.format(
            self.index,
            self.label,
            self.score
        )

    def __repr__(self):
        return self.__str__()

