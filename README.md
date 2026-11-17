This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Integración de los modelos seleccionados

`app/page.tsx` usa `public/stage1_resnet.onnx` (entrada `[1,1,4000]`) y `public/stage2_resnet (1).onnx` (entrada `[1,1,5000]`). Ambos modelos reciben señales normalizadas por ventana y devuelven logits; se aplica softmax y argmax como en Colab. Los archivos anteriores de Random Forest no se utilizan.

El runtime WASM se carga desde jsDelivr con la versión indicada por `ort.env.versions.web`; no requiere `public/ort/`.

**Pendiente de datos:** las cinco señales de `public/signals.json` tienen 4000 muestras. Para ejecutar la etapa 2 se deben regenerar las señales Chapman con al menos 5000 muestras reales a 500 Hz. La página muestra un aviso y no fabrica las muestras faltantes.

**Verificación local:** se instaló el componente nativo de Tailwind para Windows que faltaba y se corrigió la carga de ONNX Runtime: ahora se importa dinámicamente dentro de `useEffect`, solo en el navegador, evitando `ERR_INVALID_URL` durante el prerenderizado. TypeScript y `npm run build` finalizaron correctamente, incluidas las cuatro páginas estáticas. Falta comprobar la inferencia en el navegador y completar las señales de etapa 2.

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
