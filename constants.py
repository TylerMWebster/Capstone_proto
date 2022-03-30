import os
import pathlib


PATH = pathlib.Path(__file__).parent.resolve()
API_TEMP_FILENAME = "api_temp_file.csv"
SENSOR_TEMP_FILENAME = "sensor_temp.csv"
FEATURES = ["temp", "tod", "season",
            "pressure", "humidity", "dew_pt", "clouds",
            "wind_speed", "wind_deg"]


def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass
