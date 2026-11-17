This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Integración de los modelos seleccionados

`app/page.tsx` usa `public/stage1_resnet.onnx` y `public/stage2_resnet_4000.onnx`, ambos con entrada `[1,1,4000]`. Reciben señales normalizadas por ventana y devuelven logits; se aplica softmax y argmax. Los archivos anteriores de Random Forest no se utilizan.

El runtime WASM se carga desde jsDelivr con la versión indicada por `ort.env.versions.web`; no requiere `public/ort/`.

**Etapa 2 con 4000 muestras:** el original `stage2_resnet (1).onnx` se conserva sin cambios. Su arquitectura convolucional termina en `GlobalAveragePool`, `Flatten` y `Gemm`, por lo que admite la ventana más corta al adaptar la forma declarada de entrada. `node scripts/adapt-stage2.cjs` crea la variante cambiando únicamente los dos bytes que declaran la longitud 5000 por 4000; no cambia pesos ni operadores. Se usan las 4000 muestras reales (8 segundos a 500 Hz), sin relleno ni estiramiento. La precisión medida con ventanas de 5000 no puede atribuirse automáticamente a esta variante: falta evaluarla con datos etiquetados a 4000 muestras. El selector de demostración decide qué modelo ejecutar: Active usa únicamente la etapa 1; Normal, Taquicardia, Bradicardia y AFib usan directamente la etapa 2. La predicción no se fuerza a coincidir con el nombre del ejemplo.

**Verificación:** `node scripts/verify-stage2.cjs` ejecuta la variante con las cinco señales reales incluidas y comprueba que devuelve cuatro logits finitos. También comprueba que el original rechaza una entrada de 4000. Esta prueba verifica ejecución, no precisión diagnóstica. ONNX Runtime se importa dinámicamente dentro de `useEffect` para ejecutarse solo en el navegador.

## Despliegue en Vercel

- Importar `fabricioguillen5/2Parcial_IA_2026` desde la cuenta de Vercel. Si no aparece, agregar ese repositorio a la instalación de Vercel en GitHub.
- El repositorio tiene `package.json` en la raíz: seleccionar el framework Next.js y dejar Root Directory en `./`.
- Comando de compilación: `npm run build`. Usa Webpack y la integración oficial de Tailwind con PostCSS para evitar el fallo local de procesos de Turbopack.
- No se requieren claves de API ni variables de entorno: los modelos ONNX se ejecutan en el navegador. El navegador necesita acceso a jsDelivr para cargar el runtime WASM.
- Ambas etapas aceptan los ejemplos incluidos de 4000 muestras. La página identifica la adaptación y su validación de precisión pendiente.


## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
