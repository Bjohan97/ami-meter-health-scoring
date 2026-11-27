# ami-meter-health-scoring — notas técnicas en español

**Objetivo:** decidir **qué medidores visitar primero y por qué**, a partir de los huecos de comunicación,
la calidad de tensión y los eventos.

**Origen:** es la reimplementación rediseñada de una herramienta interna que desarrollé con el nombre
**"Gemelo Digital"**. Esta versión es su capa de diagnóstico y priorización; un modelo físico o de estado
de la red de medición (un gemelo digital en sentido estricto) queda como trabajo futuro.

## Por qué está diseñado así

- **Scores interpretables y separados de las features de diagnóstico.** La disponibilidad, la comparación
  con pares, la estabilidad, la tensión y los eventos se explican cada uno en una frase. La concentración
  de los huecos por día (Herfindahl normalizado), su reparto por horas (entropía de Shannon) y la
  proporción de días completos perdidos sirven para **clasificar la causa**, no para sumar puntos.
- **Un módulo sin datos no cuenta como perfecto.** Los pesos se renormalizan sobre los módulos
  disponibles, y `coverage_of_score` indica qué parte del peso total tenía datos.
- **La tensión usa solo valores válidos.** Un código de relé abierto no es una tensión, y 0 V (un corte) no
  es un problema de calidad de tensión. El factor de potencia no se usa.
- **El tamper se decide por orden temporal.** Si el último evento de tapa es "abierta", está sin resolver.
- **Primero se descarta la red y el suministro.** Si un DCU entero falla (`GROUP_ISSUE`) o hubo cortes de
  energía (`POWER_OUTAGE`), el problema no es el medidor y no hace falta visitarlo.
- **La tendencia describe el pasado; no es una predicción.**

## Cómo leer el resultado

`field_list.csv` ordena los medidores por prioridad de diagnóstico y, dentro de cada diagnóstico, por
score de salud. Cada fila trae la regla que se disparó, los números que la dispararon y una acción
sugerida.

## Limitaciones

Los umbrales son valores por defecto ilustrativos, no calibrados con inspecciones reales. Todavía no hay
un detector de anomalías con aprendizaje automático. La comparación con pares necesita un mapeo
medidor → grupo. Un diagnóstico es una ayuda para priorizar, no un certificado de avería.

---

**Autor:** Brayam Martinez · **Licencia:** [MIT](../LICENSE) · Desarrollado con asistencia de IA (Claude).
