# SafeWork

Monitoramento de EPIs em zonas de risco com OpenCV e YOLOv8.
Estado atual: **Módulo 1 concluído** (captura, inferência, ROI poligonal e análise espacial).

## Estrutura

```text
SafeWork/
├── main.py                  # linha de comando
├── requirements.txt
├── config/
│   ├── settings.yaml        # câmera, modelo, pré-processamento
│   └── zones.json           # zonas de risco (coordenadas normalizadas)
├── safework/
│   ├── app.py               # monta os objetos e conduz o laço de exibição
│   ├── core/                # modelos de domínio e configuração
│   ├── vision/              # MÓDULO 1
│   │   ├── capture.py       # webcam/RTSP/arquivo em thread, com reconexão
│   │   ├── preprocessing.py # redimensionamento e redução de ruído
│   │   ├── detector.py      # YOLOv8 + rastreamento ByteTrack
│   │   ├── roi.py           # zona poligonal, máscara e persistência
│   │   ├── roi_editor.py    # desenho das zonas com o mouse
│   │   ├── spatial.py       # ponto-no-polígono e vínculo de EPI à pessoa
│   │   ├── renderer.py      # overlays
│   │   └── pipeline.py      # orquestração das etapas
│   ├── logic/               # Módulo 2: máquina de estados (a implementar)
│   ├── audit/               # Módulo 3: evidências e auditoria (a implementar)
│   ├── notification/        # Módulo 4: e-mail SMTP (a implementar)
│   ├── gui/                 # Módulo 5: interface industrial (a implementar)
│   └── utils/               # logging e FPS
├── models/                  # pesos .pt
├── evidences/  logs/  data/
└── tests/
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py --edit-zones                    # desenha as zonas sobre um frame da câmera
python main.py                                 # monitora a fonte do settings.yaml (Q ou ESC sai)
python main.py --source data/teste.mp4         # usa um vídeo
python main.py --source "rtsp://usuario:senha@10.0.0.5/live"
python main.py --source data/teste.mp4 --headless --output logs/saida.mp4
python -m pytest                               # testes
```

## Modelo de EPI

Coloque os pesos treinados em `models/safework_ppe.pt`. Sem eles, o sistema baixa o
`yolov8n.pt` (COCO), que só detecta pessoas: quem está na zona aparece como
**EPI não verificado**, nunca como infração.

As classes são vinculadas pelo **nome**, não pelo índice. Se o dataset usar outros nomes,
acrescente-os em `detector.class_aliases` no `settings.yaml`.

## Legenda

| Cor | Situação |
| --- | --- |
| Verde | Na zona, com os EPIs exigidos |
| Vermelho | Na zona, faltando EPI |
| Amarelo | Na zona, EPI não verificável pelo modelo |
| Cinza | Fora da zona |

## Contrato com o Módulo 2

`VisionPipeline.stream()` gera um `FrameAnalysis` por frame. Cada `PersonAnalysis` traz
`track_id`, `status`, `zone_id` e `missing_ppe`. O temporizador deve usar `frame.timestamp`,
que funciona tanto ao vivo quanto em vídeo gravado.
