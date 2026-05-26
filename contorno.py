import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


class EditorContornoAla:

    def __init__(self, ruta_imagen, num_puntos_iniciales=60):
        self.ruta_imagen = ruta_imagen
        self.num_puntos = num_puntos_iniciales

        self.contorno_completo = self._extraer_contorno_puro()
        self.puntos_control = self._inicializar_puntos_uniformes()

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.axis("equal")
        self.ax.grid(True)
        self.ax.set_title(
            "Editor de Ala: [Clic Izquierdo] Agregar | [Tecla D] Eliminar cerca del cursor"
        )

        self.ax.plot(
            self.contorno_completo[:, 0],
            self.contorno_completo[:, 1],
            "g-",
            alpha=0.3,
            label="Contorno Original",
        )

        (self.linea_spline,) = self.ax.plot(
            [], [], "b-", linewidth=2, label="Spline Cúbico"
        )
        (self.puntos_graficos,) = self.ax.plot(
            [], [], "ro", markersize=6, label="Puntos de Control"
        )

        self.ax.legend()

        self.fig.canvas.mpl_connect("button_press_event", self.al_hacer_clic)
        self.fig.canvas.mpl_connect("key_press_event", self.al_presionar_tecla)

        self.actualizar_grafico()

    def _extraer_contorno_puro(self):
        img = cv2.imread(self.ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"No se pudo cargar la imagen en: {self.ruta_imagen}"
            )

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contornos, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contornos:
            raise ValueError("No se detectaron contornos.")

        ala_contorno = max(contornos, key=cv2.contourArea)
        puntos = ala_contorno.reshape(-1, 2).astype(float)

        alto_img, _ = img.shape
        puntos[:, 1] = alto_img - puntos[:, 1]
        return puntos

    def _inicializar_puntos_uniformes(self):
        total_puntos = len(self.contorno_completo)
        indices = np.linspace(
            0, total_puntos - 1, self.num_puntos, dtype=int
        )
        return list(self.contorno_completo[indices])

    def actualizar_grafico(self):
        if len(self.puntos_control) < 3:
            # El spline cúbico periódico requiere al menos 3 puntos
            self.linea_spline.set_data([], [])
            puntos_arr = np.array(self.puntos_control)
            self.puntos_graficos.set_data(puntos_arr[:, 0], puntos_arr[:, 1])
            self.fig.canvas.draw_idle()
            return

        self.ordenar_puntos()

        puntos_arr = np.array(self.puntos_control)

        puntos_cierre = np.vstack([puntos_arr, puntos_arr[0]])

        t = np.linspace(0, 1, len(puntos_cierre))
        cs_x = CubicSpline(t, puntos_cierre[:, 0], bc_type="periodic")
        cs_y = CubicSpline(t, puntos_cierre[:, 1], bc_type="periodic")

        t_fino = np.linspace(0, 1, 1000)
        x_suave = cs_x(t_fino)
        y_suave = cs_y(t_fino)

        self.linea_spline.set_data(x_suave, y_suave)
        self.puntos_graficos.set_data(puntos_arr[:, 0], puntos_arr[:, 1])
        self.fig.canvas.draw_idle()

    def ordenar_puntos(self):
        def encontrar_indice_original(p):
            distancias = np.sum((self.contorno_completo - p) ** 2, axis=1)
            return np.argmin(distancias)

        self.puntos_control.sort(key=encontrar_indice_original)

    def al_hacer_clic(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        nuevo_punto = [event.xdata, event.ydata]
        self.puntos_control.append(nuevo_punto)
        self.actualizar_grafico()

    def al_presionar_tecla(self, event):
        if event.key == "d" and event.inaxes == self.ax:
            if len(self.puntos_control) == 0:
                return

            pos_mouse = np.array([event.xdata, event.ydata])
            puntos_arr = np.array(self.puntos_control)

            distancias = np.sum((puntos_arr - pos_mouse) ** 2, axis=1)
            indice_cercano = np.argmin(distancias)

            if np.sqrt(distancias[indice_cercano]) < 30:
                self.puntos_control.pop(indice_cercano)
                self.actualizar_grafico()

if __name__ == "__main__":
    # aca reemplacen con el nombre del archivo o ruta, en este caso: ala.jpg
    archivo_imagen = "ala.jpg"

    try:
        # aca pueden cambiar cuantos puntos quieren incialmente, en este caso eleji 50 num_puntos_iniciales=50
        editor = EditorContornoAla(archivo_imagen, num_puntos_iniciales=50)
        plt.show()

        puntos_finales = np.array(editor.puntos_control)
        np.savetxt(
            "puntos.txt",
            puntos_finales,
            fmt="%.2f",
            header="X Y",
            comments="",
        )
        print(
            f"¡Guardado exitoso! Archivo 'puntos.txt' creado con {len(puntos_finales)} puntos."
        )

    except Exception as e:
        print(f"Error: {e}")
