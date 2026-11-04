"""Crea la base SQLite de la tienda ficticia "Tienda Andina" con datos reproducibles (semilla fija)."""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

RUTA_BD = Path(__file__).parent / "tienda.db"

CATEGORIAS = ["Electrónica", "Hogar", "Deportes", "Libros", "Ropa", "Juguetes", "Belleza", "Oficina"]
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Pereira"]
PRODUCTOS = [
    ("Audífonos inalámbricos", 0, 189000), ("Teclado mecánico", 0, 245000), ("Monitor 24 pulgadas", 0, 720000),
    ("Portátil 14 pulgadas", 0, 3450000), ("Mouse ergonómico", 0, 85000), ("Parlante bluetooth", 0, 159000),
    ("Licuadora", 1, 210000), ("Juego de sartenes", 1, 175000), ("Cafetera", 1, 265000), ("Lámpara de escritorio", 1, 69000),
    ("Almohada viscoelástica", 1, 98000), ("Balón de fútbol", 2, 89000), ("Bicicleta de ruta", 2, 2890000),
    ("Mancuernas 10 kg", 2, 155000), ("Tapete de yoga", 2, 72000), ("Raqueta de tenis", 2, 430000),
    ("Cien años de soledad", 3, 59000), ("Libro de Python", 3, 115000), ("Atlas de Colombia", 3, 135000),
    ("Novela gráfica", 3, 48000), ("Chaqueta impermeable", 4, 320000), ("Camiseta deportiva", 4, 65000),
    ("Jean clásico", 4, 149000), ("Tenis para correr", 4, 389000), ("Rompecabezas 1000 piezas", 5, 79000),
    ("Bloques de construcción", 5, 199000), ("Carro a control remoto", 5, 245000), ("Muñeca articulada", 5, 99000),
    ("Crema hidratante", 6, 45000), ("Perfume floral", 6, 280000), ("Secador de pelo", 6, 189000),
    ("Kit de maquillaje", 6, 135000), ("Silla ergonómica", 7, 890000), ("Resma de papel", 7, 23000),
    ("Archivador metálico", 7, 410000), ("Calculadora científica", 7, 92000),
    ("Tablet 10 pulgadas", 0, 1250000), ("Organizador de cables", 7, 35000),  # los dos últimos nunca se venden
]
NOMBRES = ["Ana", "Carlos", "Laura", "Andrés", "Sofía", "Juan", "Valentina", "Diego", "Camila", "Santiago",
           "Isabela", "Mateo", "Daniela", "Sebastián", "Mariana", "Felipe", "Paula", "Nicolás", "Gabriela", "Tomás"]
APELLIDOS = ["Gómez", "Rodríguez", "Martínez", "López", "García", "Pérez", "Sánchez", "Ramírez", "Torres", "Díaz",
             "Vargas", "Rojas", "Moreno", "Castro", "Ortiz"]


def crear(ruta: Path = RUTA_BD) -> Path:
    rng = random.Random(42)
    ruta.unlink(missing_ok=True)
    con = sqlite3.connect(ruta)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE categorias (id INTEGER PRIMARY KEY, nombre TEXT NOT NULL);
        CREATE TABLE productos (
            id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, categoria_id INTEGER REFERENCES categorias(id),
            precio INTEGER NOT NULL, stock INTEGER NOT NULL);
        CREATE TABLE clientes (
            id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, ciudad TEXT NOT NULL,
            segmento TEXT NOT NULL, fecha_registro TEXT NOT NULL);
        CREATE TABLE pedidos (
            id INTEGER PRIMARY KEY, cliente_id INTEGER REFERENCES clientes(id),
            fecha TEXT NOT NULL, estado TEXT NOT NULL);
        CREATE TABLE detalle_pedido (
            pedido_id INTEGER REFERENCES pedidos(id), producto_id INTEGER REFERENCES productos(id),
            cantidad INTEGER NOT NULL, precio_unitario INTEGER NOT NULL);
        """
    )
    cur.executemany("INSERT INTO categorias VALUES (?, ?)", [(i + 1, n) for i, n in enumerate(CATEGORIAS)])
    productos = []
    for i, (nombre, cat, precio) in enumerate(PRODUCTOS):
        stock = rng.randint(0, 60)
        productos.append((i + 1, nombre, cat + 1, precio, stock))
    cur.executemany("INSERT INTO productos VALUES (?, ?, ?, ?, ?)", productos)

    clientes, usados = [], set()
    while len(clientes) < 60:
        nombre = f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)}"
        if nombre in usados:
            continue
        usados.add(nombre)
        ciudad = rng.choices(CIUDADES, weights=[30, 22, 16, 12, 10, 10])[0]
        segmento = "premium" if rng.random() < 0.3 else "normal"
        registro = date(2024, 1, 1) + timedelta(days=rng.randint(0, 500))
        clientes.append((len(clientes) + 1, nombre, ciudad, segmento, registro.isoformat()))
    cur.executemany("INSERT INTO clientes VALUES (?, ?, ?, ?, ?)", clientes)

    vendibles = [p for p in productos if p[0] <= 36]
    pesos = [max(1, 12 - i // 3) for i in range(len(vendibles))]
    clientes_activos = [c[0] for c in clientes[:54]]  # los últimos 6 clientes nunca compran
    pedidos, detalle = [], []
    for pid in range(1, 301):
        cliente = rng.choice(clientes_activos)
        fecha = date(2025, 1, 1) + timedelta(days=rng.randint(0, 364))
        estado = rng.choices(["entregado", "pendiente", "cancelado"], weights=[75, 15, 10])[0]
        pedidos.append((pid, cliente, fecha.isoformat(), estado))
        for prod in rng.sample(rng.choices(vendibles, weights=pesos, k=6), k=rng.randint(1, 3)):
            detalle.append((pid, prod[0], rng.randint(1, 4), prod[3]))
    # dedup producto repetido dentro de un pedido
    vistos, detalle_unico = set(), []
    for d in detalle:
        if (d[0], d[1]) not in vistos:
            vistos.add((d[0], d[1]))
            detalle_unico.append(d)
    cur.executemany("INSERT INTO pedidos VALUES (?, ?, ?, ?)", pedidos)
    cur.executemany("INSERT INTO detalle_pedido VALUES (?, ?, ?, ?)", detalle_unico)
    con.commit()
    con.close()
    return ruta


if __name__ == "__main__":
    print("Base creada en", crear())
