# SQL Project: Event Management Database Design & Operations

# Introduction

This project focuses on designing and implementing a relational database for managing cultural events efficiently using SQL. The goal is to showcase database modeling, query execution, and optimization techniques to ensure smooth event management, financial tracking, and audience engagement analysis.

The system will store structured data on events, artists, venues, ticket sales, and attendees. By leveraging SQL, the database will support advanced queries, data integrity constraints, and performance optimizations to streamline event operations.

# Database Requirements

## 1. Event Management Entities

The database will store structured data on:

- Events: Name, activity type, date, time, venue, ticket price, and description.
- Artists: Name, biography, and performance fees.
- Venues: Name, location, seating capacity, rental price, and features.
- Attendees: Name, contact information, and attendance records.
- Tickets: Sales records for revenue analysis.

## 2. Relationship & Constrains 
- Each event is associated with one venue.
- Each event belongs to one activity type (e.g., concert, exhibition, theater, conference).
- Multiple artists can perform in one event.
- Attendees can participate in multiple events.
- Ticket sales should be linked to attendees and events.
- Foreign keys and constraints will ensure data integrity.

# SQL Implementation

## Database Creation

```ruby
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
```
## Example Queries
### 1. Income by cities considering multiple events and accumulated costs
```ruby
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
```

### 2. Events with more attendance averaged by the total people alowed.

```ruby
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
```

### 3. Top 3 cities with most percentage of succesful events

```ruby
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
```



