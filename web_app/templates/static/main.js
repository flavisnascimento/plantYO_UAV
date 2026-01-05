// ===================================================================
// ESTADO GLOBAL E CONFIGURAÇÕES
// ===================================================================
const State = {
  drawnPolygon: null,
  map: null,
  plantingPoints: [],
  plantingMarkers: [],
  drawingManager: null,
  mapInitialized: false,
  currentTab: 'grid',
  // Novos estados para validação
  validationCanvas: null,
  talhoes: [],
  talhaoSelecionado: null,
  parcelas: [],
  desenhando: false,
  pontoAtual: [],
  modo: 'selecionar',
  tipoVegetacao: 'cerrado'
};

const CONFIG = {
  DEFAULT_CENTER: { lat: -21.986580, lng: -47.879703 }, // UFSCar São Carlos
  DEFAULT_ZOOM: 18,
  BACKEND_URL: window.location.origin + '/',
  MAX_WAYPOINTS: 1000,
};

const SPECIES_CONFIG = {
  Ervas: ['Erva-doce', 'Camomila', 'Capim-cidreira', 'Hortelã', 'Manjericão'],
  Arbustos: ['Hibisco', 'Manacá', 'Café', 'Pitanga', 'Azaleia'],
  Árvores: ['Ipê', 'Jatobá', 'Jacarandá', 'Pau-brasil', 'Cedro']
};

// Mapa de cores para categorias
const CATEGORY_COLORS = {
  'Ervas': '#22c55e',
  'Arbustos': '#eab308',
  'Árvores': '#3b82f6'
};

// ===================================================================
// SISTEMA DE NOTIFICAÇÕES
// ===================================================================
class NotificationSystem {
  static show(message, type = 'info', duration = 5000) {
    // Criar container se não existir
    let container = document.getElementById('notification-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'notification-container';
      container.className = 'fixed top-4 right-4 z-50 space-y-2';
      document.body.appendChild(container);
    }

    const colors = {
      success: 'bg-green-500',
      error: 'bg-red-500',
      warning: 'bg-yellow-500',
      info: 'bg-blue-500'
    };

    const notification = document.createElement('div');
    notification.className = `${colors[type]} text-white p-4 rounded-lg shadow-lg transform translate-x-full transition-transform duration-300`;
    notification.innerHTML = `
      <div class="flex items-center justify-between">
        <span>${message}</span>
        <button class="ml-4 text-white hover:text-gray-200" onclick="this.parentElement.parentElement.remove()">
          ✕
        </button>
      </div>
    `;

    container.appendChild(notification);

    // Animate in
    setTimeout(() => {
      notification.classList.remove('translate-x-full');
    }, 100);

    // Auto remove
    setTimeout(() => {
      notification.classList.add('translate-x-full');
      setTimeout(() => notification.remove(), 300);
    }, duration);
  }
}

// ===================================================================
// INICIALIZAÇÃO DO MAPA
// ===================================================================
function initMap() {
  try {
    State.map = new google.maps.Map(document.getElementById('map'), {
      center: CONFIG.DEFAULT_CENTER,
      zoom: CONFIG.DEFAULT_ZOOM,
      mapTypeId: google.maps.MapTypeId.SATELLITE,
      mapTypeControl: true,
      streetViewControl: false,
      fullscreenControl: true,
      zoomControl: true
    });

    // Inicializar Drawing Manager
    State.drawingManager = new google.maps.drawing.DrawingManager({
      drawingMode: null,
      drawingControl: true,
      drawingControlOptions: {
        position: google.maps.ControlPosition.TOP_CENTER,
        drawingModes: ['polygon']
      },
      polygonOptions: {
        fillColor: '#00CED1',
        fillOpacity: 0.3,
        strokeWeight: 2,
        strokeColor: '#008B8B',
        clickable: false,
        editable: true
      }
    });

    State.drawingManager.setMap(State.map);

    // Event listeners para desenhar polígono
    google.maps.event.addListener(State.drawingManager, 'polygoncomplete', function(polygon) {
      if (State.drawnPolygon) {
        State.drawnPolygon.setMap(null);
      }
      State.drawnPolygon = polygon;
      State.drawingManager.setDrawingMode(null);
      
      // Calcular área
      const area = google.maps.geometry.spherical.computeArea(polygon.getPath());
      const areaHa = (area / 10000).toFixed(2);
      NotificationSystem.show(`Área selecionada: ${areaHa} hectares`, 'success');
    });

    // Event listener para adicionar pontos manuais
    google.maps.event.addListener(State.map, 'click', function(event) {
      if (State.currentTab === 'pontos') {
        addPlantingPoint(event.latLng);
      }
    });

    State.mapInitialized = true;
    NotificationSystem.show('Mapa inicializado com sucesso!', 'success');
    
    // Inicializar outras funcionalidades
    updateSpeciesOptions();

  } catch (error) {
    console.error('Erro ao inicializar mapa:', error);
    NotificationSystem.show('Erro ao inicializar mapa: ' + error.message, 'error');
  }
}

// ===================================================================
// SISTEMA DE ABAS
// ===================================================================
function setTab(tabName) {
  // Remover classes ativas de todos os botões
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('tab-btn-active');
    btn.classList.add('text-gray-600', 'hover:text-gray-900');
  });

  // Ocultar todas as abas
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.add('hidden');
  });

  // Ativar aba selecionada
  const activeBtn = document.querySelector(`[onclick="setTab('${tabName}')"]`);
  if (activeBtn) {
    activeBtn.classList.add('tab-btn-active');
    activeBtn.classList.remove('text-gray-600', 'hover:text-gray-900');
  }

  const tabElement = document.getElementById(`tab-${tabName}`);
  if (tabElement) {
    tabElement.classList.remove('hidden');
  }
  
  State.currentTab = tabName;

  // Configurar modo do drawing manager
  if (State.drawingManager) {
    if (tabName === 'grid') {
      State.drawingManager.setOptions({ drawingControl: true });
    } else {
      State.drawingManager.setOptions({ drawingControl: false });
      State.drawingManager.setDrawingMode(null);
    }
  }

  // Inicializar canvas se for aba validação
  if (tabName === 'validacao') {
    setTimeout(() => initValidationCanvas(), 100);
  }
}

// ===================================================================
// SISTEMA DE VALIDAÇÃO - PARCELAS DE MONITORAMENTO
// ===================================================================

// Função para calcular área de um polígono usando fórmula de Shoelace
function calcularArea(pontos) {
  if (pontos.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < pontos.length; i++) {
    const j = (i + 1) % pontos.length;
    area += pontos[i].x * pontos[j].y;
    area -= pontos[j].x * pontos[i].y;
  }
  return Math.abs(area / 2) / 10000; // Convertendo para hectares (aproximação)
}

// Função para calcular número de parcelas baseado na área
function calcularNumeroParcelas(area) {
  if (area <= 1) return 5;
  return Math.min(Math.ceil(area) + 4, 50);
}

// Função para gerar parcelas aleatórias dentro do talhão
function gerarParcelas(talhao) {
  const numParcelas = calcularNumeroParcelas(talhao.area);
  const novasParcelas = [];

  // Encontrar bounds do talhão
  const minX = Math.min(...talhao.pontos.map(p => p.x));
  const maxX = Math.max(...talhao.pontos.map(p => p.x));
  const minY = Math.min(...talhao.pontos.map(p => p.y));
  const maxY = Math.max(...talhao.pontos.map(p => p.y));

  // Função para verificar se um ponto está dentro do polígono
  const pontoNoPolgono = (ponto, pontos) => {
    let dentroDoPoligono = false;
    for (let i = 0, j = pontos.length - 1; i < pontos.length; j = i++) {
      if (((pontos[i].y > ponto.y) !== (pontos[j].y > ponto.y)) &&
          (ponto.x < (pontos[j].x - pontos[i].x) * (ponto.y - pontos[i].y) / (pontos[j].y - pontos[i].y) + pontos[i].x)) {
        dentroDoPoligono = !dentroDoPoligono;
      }
    }
    return dentroDoPoligono;
  };

  for (let i = 0; i < numParcelas; i++) {
    let tentativas = 0;
    let parcelaCentro;
    
    // Tenta encontrar um ponto válido dentro do talhão
    do {
      parcelaCentro = {
        x: minX + Math.random() * (maxX - minX),
        y: minY + Math.random() * (maxY - minY)
      };
      tentativas++;
    } while (!pontoNoPolgono(parcelaCentro, talhao.pontos) && tentativas < 50);

    if (tentativas < 50) {
      // Parcela 25m x 4m (escala aproximada)
      const parcela = {
        id: `P${i + 1}`,
        centro: parcelaCentro,
        largura: 4,
        comprimento: 25,
        angulo: Math.random() * 360, // Ângulo aleatório
        talhaoId: talhao.id
      };
      novasParcelas.push(parcela);
    }
  }

  return novasParcelas;
}

// Inicializar canvas de validação
function initValidationCanvas() {
  State.validationCanvas = document.getElementById('validation-canvas');
  if (State.validationCanvas) {
    State.validationCanvas.addEventListener('click', handleValidationCanvasClick);
    desenharValidationCanvas();
  }
}

// Desenho no canvas de validação
function desenharValidationCanvas() {
  const canvas = State.validationCanvas;
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Desenhar grid de fundo
  ctx.strokeStyle = '#f0f0f0';
  ctx.lineWidth = 1;
  for (let i = 0; i < canvas.width; i += 50) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
    ctx.stroke();
  }
  for (let i = 0; i < canvas.height; i += 50) {
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(canvas.width, i);
    ctx.stroke();
  }

  // Desenhar talhões
  State.talhoes.forEach((talhao) => {
    ctx.fillStyle = talhao.id === State.talhaoSelecionado?.id ? '#22c55e20' : '#3b82f620';
    ctx.strokeStyle = talhao.id === State.talhaoSelecionado?.id ? '#22c55e' : '#3b82f6';
    ctx.lineWidth = 2;

    if (talhao.pontos.length > 2) {
      ctx.beginPath();
      ctx.moveTo(talhao.pontos[0].x, talhao.pontos[0].y);
      for (let i = 1; i < talhao.pontos.length; i++) {
        ctx.lineTo(talhao.pontos[i].x, talhao.pontos[i].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Label do talhão
      const centroX = talhao.pontos.reduce((sum, p) => sum + p.x, 0) / talhao.pontos.length;
      const centroY = talhao.pontos.reduce((sum, p) => sum + p.y, 0) / talhao.pontos.length;
      ctx.fillStyle = '#1f2937';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${talhao.nome}`, centroX, centroY - 10);
      ctx.fillText(`${talhao.area.toFixed(2)} ha`, centroX, centroY + 5);
    }
  });

  // Desenhar parcelas
  State.parcelas.forEach((parcela) => {
    const { centro, largura, comprimento, angulo, id } = parcela;
    
    ctx.save();
    ctx.translate(centro.x, centro.y);
    ctx.rotate((angulo * Math.PI) / 180);
    
    // Retângulo da parcela
    ctx.fillStyle = '#ef4444aa';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.fillRect(-comprimento/2, -largura/2, comprimento, largura);
    ctx.strokeRect(-comprimento/2, -largura/2, comprimento, largura);
    
    // Linha amostral central
    ctx.strokeStyle = '#7c2d12';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-comprimento/2, 0);
    ctx.lineTo(comprimento/2, 0);
    ctx.stroke();
    
    // Label da parcela
    ctx.fillStyle = '#7c2d12';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(id, 0, -8);
    
    ctx.restore();
  });

  // Desenhar polígono em construção
  if (State.pontoAtual.length > 0) {
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    ctx.beginPath();
    ctx.moveTo(State.pontoAtual[0].x, State.pontoAtual[0].y);
    for (let i = 1; i < State.pontoAtual.length; i++) {
      ctx.lineTo(State.pontoAtual[i].x, State.pontoAtual[i].y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Pontos
    State.pontoAtual.forEach((ponto, index) => {
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(ponto.x, ponto.y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
}

// Event handler para cliques no canvas
function handleValidationCanvasClick(e) {
  const canvas = State.validationCanvas;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (State.modo === 'desenhar') {
    State.pontoAtual.push({ x, y });
  } else if (State.modo === 'selecionar') {
    // Verificar se clicou em um talhão
    const talhaoClicado = State.talhoes.find(talhao => {
      // Algoritmo point-in-polygon
      let dentroDoPoligono = false;
      for (let i = 0, j = talhao.pontos.length - 1; i < talhao.pontos.length; j = i++) {
        if (((talhao.pontos[i].y > y) !== (talhao.pontos[j].y > y)) &&
            (x < (talhao.pontos[j].x - talhao.pontos[i].x) * (y - talhao.pontos[i].y) / (talhao.pontos[j].y - talhao.pontos[i].y) + talhao.pontos[i].x)) {
          dentroDoPoligono = !dentroDoPoligono;
        }
      }
      return dentroDoPoligono;
    });

    State.talhaoSelecionado = talhaoClicado || null;
    updateValidationInfo();
  }
  
  desenharValidationCanvas();
}

// Funções de controle da validação
function setValidationMode(mode) {
  State.modo = mode;
  if (mode === 'desenhar') {
    State.desenhando = true;
    State.pontoAtual = [];
  }
  
  // Atualizar botões
  document.querySelectorAll('.validation-mode-btn').forEach(btn => {
    btn.classList.remove('bg-blue-600', 'text-white', 'bg-green-600');
    btn.classList.add('bg-gray-200', 'text-gray-700');
  });
  
  const activeBtn = document.querySelector(`[onclick="setValidationMode('${mode}')"]`);
  if (activeBtn) {
    activeBtn.classList.remove('bg-gray-200', 'text-gray-700');
    activeBtn.classList.add(mode === 'desenhar' ? 'bg-blue-600' : 'bg-green-600', 'text-white');
  }
}

function finalizarTalhao() {
  if (State.pontoAtual.length < 3) {
    NotificationSystem.show('Um talhão precisa ter pelo menos 3 pontos!', 'warning');
    return;
  }

  const area = calcularArea(State.pontoAtual);
  const novoTalhao = {
    id: Date.now(),
    nome: `Talhão ${State.talhoes.length + 1}`,
    pontos: [...State.pontoAtual],
    area: area,
    tipoVegetacao: State.tipoVegetacao
  };

  State.talhoes.push(novoTalhao);
  State.pontoAtual = [];
  State.desenhando = false;
  setValidationMode('selecionar');
  
  NotificationSystem.show('Talhão criado com sucesso!', 'success');
  desenharValidationCanvas();
}

function criarParcelas() {
  if (!State.talhaoSelecionado) {
    NotificationSystem.show('Selecione um talhão primeiro!', 'warning');
    return;
  }

  const novasParcelas = gerarParcelas(State.talhaoSelecionado);
  State.parcelas = State.parcelas.filter(p => p.talhaoId !== State.talhaoSelecionado.id).concat(novasParcelas);
  
  desenharValidationCanvas();
  updateValidationInfo();
  
  NotificationSystem.show(`${novasParcelas.length} parcelas criadas!`, 'success');
}

function limparValidacao() {
  State.talhoes = [];
  State.parcelas = [];
  State.talhaoSelecionado = null;
  State.pontoAtual = [];
  
  desenharValidationCanvas();
  updateValidationInfo();
  
  NotificationSystem.show('Validação limpa!', 'info');
}

function updateValidationInfo() {
  const infoElement = document.getElementById('validation-info');
  if (!infoElement) return;
  
  if (State.talhaoSelecionado) {
    const parcelasTalhao = State.parcelas.filter(p => p.talhaoId === State.talhaoSelecionado.id);
    const numParcelas = calcularNumeroParcelas(State.talhaoSelecionado.area);
    
    infoElement.innerHTML = `
      <h3 class="font-semibold text-green-800 mb-2">Talhão Selecionado:</h3>
      <p class="text-sm text-green-700"><strong>Nome:</strong> ${State.talhaoSelecionado.nome}</p>
      <p class="text-sm text-green-700"><strong>Área:</strong> ${State.talhaoSelecionado.area.toFixed(2)} ha</p>
      <p class="text-sm text-green-700"><strong>Parcelas requeridas:</strong> ${numParcelas}</p>
      <p class="text-sm text-green-700"><strong>Parcelas criadas:</strong> ${parcelasTalhao.length}</p>
    `;
  } else {
    infoElement.innerHTML = '<p class="text-gray-500">Nenhum talhão selecionado</p>';
  }
}

function exportarRelatorioValidacao() {
  if (!State.talhaoSelecionado) {
    NotificationSystem.show('Selecione um talhão primeiro!', 'warning');
    return;
  }
  
  const parcelasTalhao = State.parcelas.filter(p => p.talhaoId === State.talhaoSelecionado.id);
  const numParcelas = calcularNumeroParcelas(State.talhaoSelecionado.area);
  
  const relatorio = `RELATÓRIO DE PARCELAS DE MONITORAMENTO
Baseado na Resolução SMA 32/2014 e Portaria CBRN 01/2015

=== INFORMAÇÕES DO TALHÃO ===
Nome: ${State.talhaoSelecionado.nome}
Área: ${State.talhaoSelecionado.area.toFixed(2)} hectares
Tipo de Vegetação: ${State.talhaoSelecionado.tipoVegetacao}

=== PARCELAS GERADAS ===
Número de parcelas requeridas: ${numParcelas}
Parcelas criadas: ${parcelasTalhao.length}
Tamanho de cada parcela: 100 m² (25m x 4m)

=== METODOLOGIA ===
- Parcelas distribuídas aleatoriamente no talhão
- Cada parcela tem linha amostral de 25m no centro
- Indicadores a medir:
  1. Cobertura do solo com vegetação nativa (%)
  2. Densidade de indivíduos nativos regenerantes (ind./ha)
  3. Número de espécies nativas regenerantes

=== CRITÉRIOS DE MEDIÇÃO ===
- Altura ≥ 50 cm e CAP < 15 cm
- Apenas espécies nativas lenhosas (arbustivas/arbóreas)
- Lista única de espécies para todo o talhão

Data: ${new Date().toLocaleDateString('pt-BR')}
`;

  const blob = new Blob([relatorio], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Relatorio_Parcelas_${State.talhaoSelecionado.nome.replace(/\s+/g, '_')}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  NotificationSystem.show('Relatório exportado com sucesso!', 'success');
}

// ===================================================================
// GERAÇÃO DE GRID VIA BACKEND + GeoJSON
// ===================================================================
async function generateGrid() {
  if (!State.drawnPolygon) {
    NotificationSystem.show('Desenhe um polígono primeiro!', 'warning');
    return false;
  }

  // Extrai o polígono desenhado
  const polygon = State.drawnPolygon.getPath().getArray().map(ll => ({
    lat: ll.lat(),
    lng: ll.lng()
  }));

  // Parâmetros do usuário
  const spacingX = parseFloat(document.getElementById('spacingX').value);
  const spacingY = parseFloat(document.getElementById('spacingY').value);
  const species = {
    ervas: document.getElementById('ervas-select').value,
    arbustos: document.getElementById('arbustos-select').value,
    arvores: document.getElementById('arvores-select').value
  };

  try {
    NotificationSystem.show('Gerando grid no servidor...', 'info');

    const res = await fetch('/generate_grid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ polygon, spacingX, spacingY, species })
    });
    
    if (!res.ok) {
      throw new Error(`Erro HTTP: ${res.status}`);
    }
    
    const { waypoints } = await res.json();

    // Atualiza estado
    State.plantingPoints = waypoints;

    // Limpa camadas anteriores
    State.map.data.forEach(f => State.map.data.remove(f));

    // Monta GeoJSON
    const features = waypoints.map(pt => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [pt.lng, pt.lat]
      },
      properties: {
        category: pt.category
      }
    }));
    const geojson = { type: 'FeatureCollection', features };

    // Adiciona ao mapa
    State.map.data.addGeoJson(geojson);

    // Estiliza
    State.map.data.setStyle(feature => ({
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        fillColor: CATEGORY_COLORS[feature.getProperty('category')],
        fillOpacity: 0.8,
        strokeColor: '#ffffff',
        strokeWeight: 2,
        scale: 8
      }
    }));

    // Atualiza contador
    updatePointCount();
    NotificationSystem.show(`Grid gerado: ${waypoints.length} pontos`, 'success');

  } catch (err) {
    console.error('Erro ao gerar grid:', err);
    NotificationSystem.show('Erro ao gerar grid: ' + err.message, 'error');
    return false;
  }

  return true;
}

// ===================================================================
// PLANTIO POR PONTOS MANUAIS
// ===================================================================
function updateSpeciesOptions() {
  const categorySelect = document.getElementById('category-select');
  const speciesSelect = document.getElementById('species-select');
  
  if (!categorySelect || !speciesSelect) return;
  
  const category = categorySelect.value;
  
  speciesSelect.innerHTML = '';
  SPECIES_CONFIG[category].forEach(species => {
    const option = document.createElement('option');
    option.value = species;
    option.textContent = species;
    speciesSelect.appendChild(option);
  });
}

function addPlantingPoint(latLng) {
  const categorySelect = document.getElementById('category-select');
  const speciesSelect = document.getElementById('species-select');
  
  if (!categorySelect || !speciesSelect) return;
  
  const category = categorySelect.value;
  const species = speciesSelect.value;

  const marker = new google.maps.Marker({
    position: latLng,
    map: State.map,
    title: `${species} (${category})`,
    icon: {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: CATEGORY_COLORS[category],
      fillOpacity: 0.8,
      strokeColor: '#ffffff',
      strokeWeight: 2,
      scale: 10
    }
  });

  const point = {
    lat: latLng.lat(),
    lng: latLng.lng(),
    species: species,
    category: category
  };

  State.plantingPoints.push(point);
  State.plantingMarkers.push(marker);

  // Atualizar contador
  updatePointCount();

  NotificationSystem.show(`Ponto adicionado: ${species}`, 'success', 2000);
}

function clearPoints() {
  State.plantingPoints = [];
  State.plantingMarkers.forEach(marker => marker.setMap(null));
  State.plantingMarkers = [];
  State.map.data.forEach(feature => {
    State.map.data.remove(feature);
  });
  updatePointCount();
  
  NotificationSystem.show('Pontos limpos!', 'info');
}

function updatePointCount() {
  const countElement = document.getElementById('point-count');
  if (countElement) {
    countElement.textContent = State.plantingPoints.length;
  }
}


// ===================================================================
// SALVAR PROJETO
// ===================================================================
function saveProject() {
  const points = State.plantingPoints;
  if (!points || points.length === 0) {
    NotificationSystem.show('Nenhum ponto para salvar!', 'warning');
    return;
  }

  const project = {
    timestamp: new Date().toISOString(),
    points: points,
    totalPoints: points.length,
    polygon: State.drawnPolygon ? State.drawnPolygon.getPath().getArray().map(ll => ({
      lat: ll.lat(),
      lng: ll.lng()
    })) : null
  };

  const blob = new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `projeto_plantio_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  NotificationSystem.show('Projeto salvo com sucesso!', 'success');
}

// ===================================================================
// DOWNLOAD KML
// ===================================================================
function downloadKML() {
  const points = State.plantingPoints;
  if (!points || points.length === 0) {
    NotificationSystem.show('Nenhum ponto para exportar!', 'warning');
    return;
  }

  let kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Projeto de Plantio</name>
  <description>Gerado em ${new Date().toLocaleString('pt-BR')}</description>`;

  // Estilos
  const styles = {
    'Ervas': { color: '22c55e' },
    'Arbustos': { color: 'eab308' },
    'Árvores': { color: '3b82f6' }
  };
  
  for (const [cat, st] of Object.entries(styles)) {
    kml += `
  <Style id="${cat}">
    <IconStyle>
      <color>ff${st.color}</color>
      <scale>1.2</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href></Icon>
    </IconStyle>
  </Style>`;
  }

  // Pontos
  points.forEach((pt, i) => {
    kml += `
  <Placemark>
    <name>${pt.species}</name>
    <description>${pt.category}</description>
    <styleUrl>#${pt.category}</styleUrl>
    <Point>
      <coordinates>${pt.lng},${pt.lat},0</coordinates>
    </Point>
  </Placemark>`;
  });

  kml += `
</Document>
</kml>`;

  const blob = new Blob([kml], { type: 'application/vnd.google-earth.kml+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `projeto_plantio_${Date.now()}.kml`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  NotificationSystem.show('Arquivo KML baixado!', 'success');
}

// ===================================================================
// LIMPAR POLÍGONO
// ===================================================================
function clearPolygon() {
  if (State.drawnPolygon) {
    State.drawnPolygon.setMap(null);
    State.drawnPolygon = null;
  } else {
    NotificationSystem.show('Nenhum polígono para limpar.', 'warning');
    return;
  }

  // Limpa todos os pontos gerados no grid (Data Layer)
  State.map.data.forEach(feature => {
    State.map.data.remove(feature);
  });

  // Reseta o vetor e o contador
  State.plantingPoints = [];
  updatePointCount();

  NotificationSystem.show('Polígono e pontos removidos.', 'info');
}

// ===================================================================
// IMPORTAR KML
// ===================================================================
function importKML(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    try {
      const parser = new DOMParser();
      const kmlDoc = parser.parseFromString(e.target.result, 'application/xml');
      const placemarks = kmlDoc.getElementsByTagName('Placemark');
      let imported = 0;

      for (let i = 0; i < placemarks.length; i++) {
        const pm = placemarks[i];
        const nameTag = pm.getElementsByTagName('name')[0];
        const descTag = pm.getElementsByTagName('description')[0];
        const coordTag = pm.getElementsByTagName('coordinates')[0];
        
        if (!(nameTag && descTag && coordTag)) continue;

        const species = nameTag.textContent.trim();
        const category = descTag.textContent.trim();
        const coordText = coordTag.textContent.trim();
        const coords = coordText.split(',');
        
        if (coords.length < 2) continue;
        
        const lng = parseFloat(coords[0]);
        const lat = parseFloat(coords[1]);
        
        if (isNaN(lat) || isNaN(lng)) continue;

        // Adiciona marcador
        const marker = new google.maps.Marker({
          position: { lat, lng },
          map: State.map,
          title: `${species} (${category})`,
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: CATEGORY_COLORS[category] || '#3b82f6',
            fillOpacity: 0.8,
            strokeColor: '#ffffff',
            strokeWeight: 2,
            scale: 8
          }
        });

        State.plantingPoints.push({ lat, lng, species, category });
        State.plantingMarkers.push(marker);
        imported++;
      }

      event.target.value = '';
      updatePointCount();
      NotificationSystem.show(`Importados ${imported} pontos do KML`, 'success');
    } catch (error) {
      console.error('Erro ao importar KML:', error);
      NotificationSystem.show('Erro ao importar KML: ' + error.message, 'error');
    }
  };
  reader.readAsText(file);
}

// ===================================================================
// INICIALIZAÇÃO
// ===================================================================
document.addEventListener('DOMContentLoaded', function() {
  console.log('Sistema de planejamento de plantio inicializado');
  
  // Aguardar carregamento da API do Google Maps
  if (typeof google !== 'undefined' && google.maps) {
    initMap();
  } else {
    // Aguardar API carregar
    window.initMap = initMap;
  }

  // Event listeners
  const kmlUpload = document.getElementById('kml-upload');
  if (kmlUpload) {
    kmlUpload.addEventListener('change', importKML);
  }
});

// ===================================================================
// FUNÇÕES GLOBAIS EXPOSTAS
// ===================================================================
window.setTab = setTab;
window.updateSpeciesOptions = updateSpeciesOptions;
window.clearPoints = clearPoints;
window.saveProject = saveProject;
window.downloadKML = downloadKML;
window.clearPolygon = clearPolygon;
window.generateGrid = generateGrid;
