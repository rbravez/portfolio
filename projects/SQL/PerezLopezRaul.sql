-- Eliminar la base de datos si ya existe y crearla nuevamente
DROP DATABASE IF EXISTS ArteVidaCultural;
CREATE DATABASE ArteVidaCultural;
USE ArteVidaCultural;

-- ---------------------------------------------------------------------------------
-- Definición de la estructura de la base de datos
-- ---------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS Actividad (
    id_actividad INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    tipo VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS Artista (
    id_artista INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    biografia TEXT
);

CREATE TABLE IF NOT EXISTS Ubicacion (
    id_ubicacion INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    direccion TEXT NOT NULL,
    ciudad VARCHAR(100),
    aforo INT NOT NULL,
    precio_alquiler DECIMAL(10, 2),
    caracteristicas TEXT
);

CREATE TABLE IF NOT EXISTS Evento (
    id_evento INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    id_actividad INT,
    id_ubicacion INT,
    precio_entrada DECIMAL(10, 2) NOT NULL,
    fecha DATE NOT NULL,
    hora TIME NOT NULL,
    descripcion TEXT,
    FOREIGN KEY (id_actividad) REFERENCES Actividad(id_actividad),
    FOREIGN KEY (id_ubicacion) REFERENCES Ubicacion(id_ubicacion)
);

CREATE TABLE IF NOT EXISTS Asistente (
    id_asistente INT AUTO_INCREMENT PRIMARY KEY,
    nombre_completo VARCHAR(255) NOT NULL,
    telefono VARCHAR(15) NOT NULL,
    email VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS Evento_Asistente (
    id_evento INT,
    id_asistente INT,
    PRIMARY KEY (id_evento, id_asistente),
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento),
    FOREIGN KEY (id_asistente) REFERENCES Asistente(id_asistente)
);

CREATE TABLE IF NOT EXISTS Actividad_Artista (
    id_actividad INT,
    id_artista INT,
    cache DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (id_actividad, id_artista),
    FOREIGN KEY (id_actividad) REFERENCES Actividad(id_actividad),
    FOREIGN KEY (id_artista) REFERENCES Artista(id_artista)
);

-- ---------------------------------------------------------------------------------
-- Inserción de datos
-- ---------------------------------------------------------------------------------

-- Actividades
INSERT INTO Actividad (nombre, tipo) VALUES 
('Concierto de Jazz', 'Concierto'),
('Exposición de Arte Moderno', 'Exposición'),
('Charla sobre Inteligencia Artificial', 'Charla informativa'),
('Festival de Cine Independiente', 'Festival de Cine');

-- Artistas
INSERT INTO Artista (nombre, biografia) VALUES 
('The Rolling Notes', 'Banda de rock popular en la escena underground.'),
('Orquesta Sinfónica Metropolitana', 'Conjunto de músicos dedicados a conciertos de música clásica.'),
('Camila Torres', 'Violinista reconocida por sus interpretaciones de Bach.'),
('Laura Castillo', 'Directora de cine independiente ganadora de varios premios.'),
('Dr. Javier Morales', 'Especialista en Inteligencia Artificial con múltiples publicaciones.'),
('Luz Marina', 'Artista visual enfocada en el arte contemporáneo.');

-- Ubicaciones
INSERT INTO Ubicacion (nombre, direccion, ciudad, aforo, precio_alquiler, caracteristicas) VALUES 
('Teatro Real', 'Plaza de Isabel II, s/n', 'Madrid', 1700, 6000, 'Teatro para óperas y conciertos.'),
('Auditorio Nacional', 'Calle Príncipe de Vergara, 146', 'Madrid', 2300, 5000, 'Auditorio para conciertos.'),
('Parque Central', 'Calle Principal s/n', 'Sevilla', 5000, 2000, 'Zona al aire libre para festivales.'),
('Palacio de Congresos', 'Avenida de la Constitución, s/n', 'Sevilla', 1200, 4500, 'Lugar amplio y moderno.'),
('Sala Independiente', 'Calle del Arte, 20', 'Barcelona', 200, 1500, 'Espacio para exposiciones y charlas.'),
('Sala Magna', 'Calle Universitaria, 10', 'Barcelona', 1500, 4000, 'Sala moderna con equipo audiovisual.');

-- Eventos
INSERT INTO Evento (nombre, id_actividad, id_ubicacion, precio_entrada, fecha, hora, descripcion) VALUES 
('Concierto de Jazz en el Teatro Real', 1, 1, 40, '2024-12-01', '20:00:00', 'Un concierto de jazz con artistas de renombre.'),
('Exposición de Arte Moderno en Barcelona', 2, 5, 20, '2024-11-22', '11:00:00', 'Exposición de arte con Luz Marina.'),
('Charla sobre IA con Javier Morales', 3, 2, 30, '2024-12-02', '10:00:00', 'Charla informativa sobre IA y sus aplicaciones.'),
('Festival de Cine en Sevilla', 4, 3, 50, '2024-11-25', '18:00:00', 'Proyección de películas con Laura Castillo.'),
('Concierto Clásico: Camila Torres', 1, 2, 60, '2024-12-10', '19:00:00', 'Interpretaciones de Bach y Vivaldi.'),
('Concierto de Rock en el Parque Central', 1, 3, 30, '2024-12-05', '21:00:00', 'Noche de rock con The Rolling Notes.'),
('Exposición de Arte Moderno en Sevilla', 2, 4, 20, '2024-11-28', '09:00:00', 'Arte contemporáneo en el Palacio de Congresos.'),
('Charla IA en Barcelona', 3, 6, 30, '2024-12-03', '10:00:00', 'Charla sobre el impacto de la IA.'),
('Festival de Cine en Barcelona', 4, 6, 40, '2024-12-18', '18:00:00', 'Proyección de cine independiente.');

-- Asistentes
INSERT INTO Asistente (nombre_completo, telefono, email) VALUES 
('Carlos González', '600123456', 'carlos@gmail.com'),
('Ana Martín', '600234567', 'ana@gmail.com'),
('Luis Fernández', '600345678', 'luis@gmail.com'),
('Sofía Ramos', '600456789', 'sofia@gmail.com'),
('Juan Pérez', '600567890', 'juan@gmail.com'),
('Marta García', '600678901', 'marta@gmail.com'),
('Pedro Sánchez', '600789012', 'pedro@gmail.com'),
('Diego Álvarez', '600888999', 'diego.alvarez@gmail.com'),
('Sandra López', '600999000', 'sandra.lopez@gmail.com'),
('Isabel Herrera', '601111222', 'isabel.herrera@gmail.com'),
('Pablo Pérez', '601222333', 'pablo.perez@gmail.com'),
('Adriana Castillo', '601333444', 'adriana.castillo@gmail.com'),
('Javier Santos', '601444555', 'javier.santos@gmail.com'),
('María Vega', '601555666', 'maria.vega@gmail.com'),
('Esteban Torres', '601666777', 'esteban.torres@gmail.com'),
('Elena Gómez', '601777888', 'elena.gomez@gmail.com'),
('Raúl Moreno', '601888999', 'raul.moreno@gmail.com'),
('Carmen Ruiz', '601999000', 'carmen.ruiz@gmail.com'),
('Iván Martínez', '602000111', 'ivan.martinez@gmail.com'),
('Lorena Sánchez', '602111222', 'lorena.sanchez@gmail.com');

-- Evento_Asistente
INSERT INTO Evento_Asistente (id_evento, id_asistente) VALUES 
(1, 1), (1, 2), (1, 3), (1, 4),
(2, 5), (2, 6), (2, 7), (2, 8),
(3, 9), (3, 10), (3, 11), (3, 12),
(4, 13), (4, 14), (4, 15), (4, 16),
(5, 17), (5, 18), (5, 19), (5, 20),
(6, 1), (6, 6), (6, 11), (6, 16),
(7, 2), (7, 7), (7, 12), (7, 17),
(8, 3), (8, 8), (8, 13), (8, 18),
(9, 4), (9, 9), (9, 14), (9, 19);

-- Actividad_Artista (Distribución múltiple)
INSERT INTO Actividad_Artista (id_actividad, id_artista, cache) VALUES 
(1, 1, 5000), 
(1, 2, 4000), 
(1, 3, 4000), 
(2, 4, 4500), 
(2, 6, 3000), 
(3, 5, 3500), 
(3, 2, 4000), 
(4, 4, 4500), 
(4, 6, 3000), 
(2, 1, 4000);


-- ---------------------------------------------------------------------------------
-- Consultas simples
-- ---------------------------------------------------------------------------------

-- Eventos por tipo de actividad
SELECT tipo, COUNT(*) AS total_eventos
FROM Actividad
JOIN Evento ON Actividad.id_actividad = Evento.id_actividad
GROUP BY tipo;

-- Ciudad con más eventos
SELECT ciudad, COUNT(*) AS total_eventos
FROM Ubicacion
JOIN Evento ON Ubicacion.id_ubicacion = Evento.id_ubicacion
GROUP BY ciudad
ORDER BY total_eventos DESC
LIMIT 1;

-- Fecha con más eventos
SELECT fecha, COUNT(*) AS total_eventos
FROM Evento
GROUP BY fecha
ORDER BY total_eventos DESC
LIMIT 5;

-- Lista de asistentes a un evento
SELECT nombre_completo, email
FROM Asistente
JOIN Evento_Asistente ON Asistente.id_asistente = Evento_Asistente.id_asistente
WHERE id_evento = 1;

-- ---------------------------------------------------------------------------------
-- Consultas avanzadas
-- ---------------------------------------------------------------------------------

-- Ganancias por ciudad considerando múltiples eventos y costos acumulados
SELECT 
    U.ciudad,
    SUM(E.precio_entrada * IFNULL(EA.total_asistentes, 0)) AS ingresos_totales,
    SUM(U.precio_alquiler + IFNULL(A.cache_total, 0)) AS costos_totales,
    (SUM(E.precio_entrada * IFNULL(EA.total_asistentes, 0)) - SUM(U.precio_alquiler + IFNULL(A.cache_total, 0))) AS ganancias_netas
FROM Ubicacion U
LEFT JOIN Evento E ON U.id_ubicacion = E.id_ubicacion
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
LEFT JOIN (
    SELECT id_actividad, SUM(cache) AS cache_total
    FROM Actividad_Artista
    GROUP BY id_actividad
) A ON E.id_actividad = A.id_actividad
GROUP BY U.ciudad
ORDER BY ganancias_netas DESC;

-- Eventos con mayor asistencia en relación con el aforo
SELECT 
    E.nombre AS evento,
    U.nombre AS ubicacion,
    IFNULL(EA.total_asistentes, 0) AS total_asistentes,
    U.aforo,
    ROUND((IFNULL(EA.total_asistentes, 0) * 100.0 / U.aforo), 2) AS porcentaje_aforo_usado
FROM Evento E
JOIN Ubicacion U ON E.id_ubicacion = U.id_ubicacion
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
ORDER BY porcentaje_aforo_usado DESC
LIMIT 5;

-- Artistas más rentables considerando su caché y las ganancias generadas por sus actividades
SELECT 
    AR.nombre AS artista,
    SUM(E.precio_entrada * EA.total_asistentes) AS ingresos_generados,
    SUM(AA.cache) AS costo_artistas,
    (SUM(E.precio_entrada * EA.total_asistentes) - SUM(AA.cache)) AS rentabilidad
FROM Actividad_Artista AA
JOIN Artista AR ON AA.id_artista = AR.id_artista
JOIN Actividad AC ON AA.id_actividad = AC.id_actividad
JOIN Evento E ON AC.id_actividad = E.id_actividad
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
GROUP BY AR.id_artista
ORDER BY rentabilidad DESC
LIMIT 5;

-- Mes del año con mayores ingresos acumulados por eventos
SELECT 
    MONTH(E.fecha) AS mes,
    SUM(E.precio_entrada * EA.total_asistentes) AS ingresos_totales
FROM Evento E
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
GROUP BY MONTH(E.fecha)
ORDER BY ingresos_totales DESC;

-- Promedio de ingresos por tipo de actividad
SELECT 
    AC.tipo AS tipo_actividad,
    ROUND(SUM(E.precio_entrada * IFNULL(EA.total_asistentes, 0)) / COUNT(E.id_evento), 2) AS ingreso_promedio
FROM Actividad AC
JOIN Evento E ON AC.id_actividad = E.id_actividad
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
GROUP BY AC.tipo
ORDER BY ingreso_promedio DESC;

-- Top 3 ciudades con mayor porcentaje de eventos exitosos
SELECT 
    U.ciudad,
    COUNT(E.id_evento) AS total_eventos,
    ROUND((SUM(CASE WHEN (IFNULL(EA.total_asistentes, 0) * 100.0 / U.aforo) > 75 THEN 1 ELSE 0 END) * 100.0 / COUNT(E.id_evento)), 2) AS porcentaje_exitosos
FROM Ubicacion U
JOIN Evento E ON U.id_ubicacion = E.id_ubicacion
LEFT JOIN (
    SELECT id_evento, COUNT(id_asistente) AS total_asistentes
    FROM Evento_Asistente
    GROUP BY id_evento
) EA ON E.id_evento = EA.id_evento
GROUP BY U.ciudad
ORDER BY porcentaje_exitosos DESC
LIMIT 3;

-- Artistas con actividades en múltiples tipos de eventos
SELECT 
    A.nombre AS artista,
    COUNT(DISTINCT AC.tipo) AS tipos_de_actividades,
    GROUP_CONCAT(DISTINCT AC.tipo) AS actividades
FROM Artista A
JOIN Actividad_Artista AA ON A.id_artista = AA.id_artista
JOIN Actividad AC ON AA.id_actividad = AC.id_actividad
GROUP BY A.id_artista, A.nombre
HAVING tipos_de_actividades > 1
ORDER BY tipos_de_actividades DESC;

-- Asistentes más frecuentes en eventos de un tipo específico (ejemplo: 'Concierto')
SELECT 
    A.nombre_completo AS asistente,
    COUNT(E.id_evento) AS total_eventos_asistidos,
    AC.tipo AS tipo_actividad
FROM Asistente A
JOIN Evento_Asistente EA ON A.id_asistente = EA.id_asistente
JOIN Evento E ON EA.id_evento = E.id_evento
JOIN Actividad AC ON E.id_actividad = AC.id_actividad
WHERE AC.tipo = 'Concierto'
GROUP BY A.id_asistente, A.nombre_completo, AC.tipo
ORDER BY total_eventos_asistidos DESC
LIMIT 5;
