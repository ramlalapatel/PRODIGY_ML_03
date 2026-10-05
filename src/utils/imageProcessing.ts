export interface ProcessedImageData {
  originalUrl: string;
  grayCanvasUrl: string;
  hogCanvasUrl: string;
  width: number;
  height: number;
  grayscalePixels: number[];
  hogVector: number[];
  featureCount: number;
  decisionScore: number;
  predictedLabel: 'Cat' | 'Dog';
  confidence: number;
}

/**
 * Resizes an image and extracts grayscale and HOG representations on an HTML Canvas.
 */
export async function processImageForSVM(
  imageSource: string | HTMLImageElement,
  imgSize: number = 64,
  featureType: 'hog' | 'flatten' = 'hog'
): Promise<ProcessedImageData> {
  const img = await loadImage(imageSource);

  // 1. Grayscale Canvas (imgSize x imgSize)
  const grayCanvas = document.createElement('canvas');
  grayCanvas.width = imgSize;
  grayCanvas.height = imgSize;
  const grayCtx = grayCanvas.getContext('2d')!;

  // Draw scaled image
  grayCtx.drawImage(img, 0, 0, imgSize, imgSize);
  const imgData = grayCtx.getImageData(0, 0, imgSize, imgSize);
  const data = imgData.data;

  const grayscalePixels: number[] = new Array(imgSize * imgSize);
  for (let i = 0; i < data.length; i += 4) {
    // Standard Luminance conversion: Y = 0.299 R + 0.587 G + 0.114 B
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;
    grayscalePixels[i / 4] = gray / 255.0; // Normalized [0, 1]
  }
  grayCtx.putImageData(imgData, 0, 0);
  const grayCanvasUrl = grayCanvas.toDataURL('image/png');

  // 2. Compute HOG (Histogram of Oriented Gradients)
  const cellSize = 8;
  const numBins = 9;
  const numCellsX = Math.floor(imgSize / cellSize);
  const numCellsY = Math.floor(imgSize / cellSize);

  // Compute gradients: Gx and Gy using [-1, 0, 1]
  const gradMag: number[][] = Array.from({ length: imgSize }, () => new Array(imgSize).fill(0));
  const gradAngle: number[][] = Array.from({ length: imgSize }, () => new Array(imgSize).fill(0));

  for (let y = 1; y < imgSize - 1; y++) {
    for (let x = 1; x < imgSize - 1; x++) {
      const idx = (y * imgSize + x);
      const left = grayscalePixels[y * imgSize + (x - 1)];
      const right = grayscalePixels[y * imgSize + (x + 1)];
      const top = grayscalePixels[(y - 1) * imgSize + x];
      const bottom = grayscalePixels[(y + 1) * imgSize + x];

      const gx = right - left;
      const gy = bottom - top;
      const mag = Math.sqrt(gx * gx + gy * gy);
      let angle = Math.atan2(gy, gx) * (180 / Math.PI);
      if (angle < 0) angle += 180; // Unsigned gradients [0, 180)
      if (angle >= 180) angle = 179.99;

      gradMag[y][x] = mag;
      gradAngle[y][x] = angle;
    }
  }

  // Cell Histograms: [numCellsY][numCellsX][numBins]
  const cellHistograms: number[][][] = Array.from({ length: numCellsY }, () =>
    Array.from({ length: numCellsX }, () => new Array(numBins).fill(0))
  );

  const binWidth = 180 / numBins; // 20 degrees per bin

  for (let cy = 0; cy < numCellsY; cy++) {
    for (let cx = 0; cx < numCellsX; cx++) {
      const startX = cx * cellSize;
      const startY = cy * cellSize;

      for (let y = startY; y < startY + cellSize; y++) {
        for (let x = startX; x < startX + cellSize; x++) {
          const mag = gradMag[y][x];
          const angle = gradAngle[y][x];
          const binIndex = Math.min(Math.floor(angle / binWidth), numBins - 1);
          cellHistograms[cy][cx][binIndex] += mag;
        }
      }
    }
  }

  // 3. Render HOG Visualizer Canvas (dark slate backdrop with cyan/amber orientation vector stars)
  const hogCanvas = document.createElement('canvas');
  const scaleVisual = 3; // Upscale for crisp display (e.g. 192x192)
  hogCanvas.width = imgSize * scaleVisual;
  hogCanvas.height = imgSize * scaleVisual;
  const hogCtx = hogCanvas.getContext('2d')!;

  // Background
  hogCtx.fillStyle = '#0f172a';
  hogCtx.fillRect(0, 0, hogCanvas.width, hogCanvas.height);

  // Subtle cell grid
  hogCtx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
  hogCtx.lineWidth = 1;
  for (let i = 0; i <= numCellsX; i++) {
    hogCtx.beginPath();
    hogCtx.moveTo(i * cellSize * scaleVisual, 0);
    hogCtx.lineTo(i * cellSize * scaleVisual, hogCanvas.height);
    hogCtx.stroke();
  }
  for (let j = 0; j <= numCellsY; j++) {
    hogCtx.beginPath();
    hogCtx.moveTo(0, j * cellSize * scaleVisual);
    hogCtx.lineTo(hogCanvas.width, j * cellSize * scaleVisual);
    hogCtx.stroke();
  }

  // Draw starburst lines for each bin
  const hogVector: number[] = [];
  const maxLineLen = (cellSize * scaleVisual * 0.45);

  for (let cy = 0; cy < numCellsY; cy++) {
    for (let cx = 0; cx < numCellsX; cx++) {
      const centerX = (cx + 0.5) * cellSize * scaleVisual;
      const centerY = (cy + 0.5) * cellSize * scaleVisual;
      const hist = cellHistograms[cy][cx];
      const maxVal = Math.max(...hist, 0.001);

      for (let b = 0; b < numBins; b++) {
        const val = hist[b];
        hogVector.push(val);
        if (val > 0.05 * maxVal) {
          const normWeight = val / maxVal;
          const theta = (b * binWidth + binWidth / 2) * (Math.PI / 180);
          const dx = Math.cos(theta) * maxLineLen * normWeight;
          const dy = Math.sin(theta) * maxLineLen * normWeight;

          hogCtx.strokeStyle = `rgba(56, 189, 248, ${Math.min(1, 0.3 + normWeight * 0.7)})`;
          hogCtx.lineWidth = 1.5;
          hogCtx.beginPath();
          hogCtx.moveTo(centerX - dx, centerY - dy);
          hogCtx.lineTo(centerX + dx, centerY + dy);
          hogCtx.stroke();
        }
      }
    }
  }

  const hogCanvasUrl = hogCanvas.toDataURL('image/png');

  // Decision score simulation based on HOG visual signature
  // Dogs typically exhibit broad muzzle horizontal gradient features and snout density,
  // Cats exhibit high-frequency vertical pointed ear gradients and whiskers.
  let topVerticalGradients = 0; // Cats have sharp vertical ears (bins near 90°)
  let middleHorizontalGradients = 0; // Dogs have wide horizontal snout gradients (bins near 0° and 180°)

  for (let cy = 0; cy < numCellsY; cy++) {
    for (let cx = 0; cx < numCellsX; cx++) {
      const hist = cellHistograms[cy][cx];
      // Top half of image (ears region)
      if (cy < numCellsY / 2) {
        topVerticalGradients += hist[4]; // 80°-100° (vertical)
      }
      // Middle region (snout/muzzle)
      if (cy >= numCellsY / 3 && cy < (2 * numCellsY) / 3) {
        middleHorizontalGradients += (hist[0] + hist[8]); // 0-20° & 160-180° (horizontal)
      }
    }
  }

  // Calculate simulated SVM decision margin
  const rawDiff = middleHorizontalGradients * 1.15 - topVerticalGradients;
  const decisionScore = Number((rawDiff * 1.4).toFixed(2));
  const isDog = decisionScore > 0;
  const predictedLabel: 'Cat' | 'Dog' = isDog ? 'Dog' : 'Cat';
  const confidence = Number((0.65 + Math.min(0.32, Math.abs(decisionScore) * 0.12)).toFixed(2));

  return {
    originalUrl: typeof imageSource === 'string' ? imageSource : imageSource.src,
    grayCanvasUrl,
    hogCanvasUrl,
    width: img.naturalWidth || imgSize,
    height: img.naturalHeight || imgSize,
    grayscalePixels,
    hogVector,
    featureCount: featureType === 'hog' ? hogVector.length : grayscalePixels.length,
    decisionScore,
    predictedLabel,
    confidence,
  };
}

function loadImage(source: string | HTMLImageElement): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (source instanceof HTMLImageElement && source.complete && source.naturalWidth > 0) {
      resolve(source);
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error('Failed to load image for processing: ' + e));
    img.src = typeof source === 'string' ? source : source.src;
  });
}
