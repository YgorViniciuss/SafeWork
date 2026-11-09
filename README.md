# SafeWork

Monitoramento de EPIs em zonas de risco com OpenCV e YOLOv8.
Estado atual: **Módulos 1 e 2 concluídos** (visão computacional; máquina de estados com alertas visual e sonoro).

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
│   ├── logic/               # MÓDULO 2
│   │   ├── states.py        # AlertState (0/1/2), AlertEvent, AlertSnapshot
│   │   ├── state_machine.py # máquina de um operador: timer de 5 s e histerese
│   │   ├── engine.py        # uma máquina por ID do ByteTrack
│   │   └── events.py        # barramento de eventos (Observer) e log
│   ├── audit/               # Módulo 3: evidências e auditoria (a implementar)
│   ├── notification/        # MÓDULO 2 (alertas locais); e-mail SMTP no Módulo 4
│   │   ├── visual.py        # overlay piscante, contagem regressiva, faixa de alerta
│   │   └── sound.py         # alarme em thread: winsound, pygame ou sinal do terminal
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
python main.py --mute                          # sem alarme sonoro
python main.py --warning-seconds 3             # outro tempo de confirmação
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

## Máquina de estados (Módulo 2)

| Estado | Condição | Alerta |
| --- | --- | --- |
| 0 OK | Conforme, fora da zona ou EPI não verificável | Nenhum |
| 1 Infração inicial | Na zona sem EPI; inicia o timer de 5 s | Caixa e borda piscando vermelho/amarelo, contagem regressiva, bipe intermitente |
| 2 Infração confirmada | Ainda sem EPI após 5 s | Caixa e borda vermelhas, faixa de infração, bipe agudo; evento `VIOLATION_CONFIRMED` |

- O timer é por operador (ID do ByteTrack) e usa o tempo do frame.
- Volta ao Estado 0 só após `clear_after_s` (1 s) de conformidade contínua; um frame isolado com capacete não zera o timer.
- Oclusões menores que `lost_track_timeout_s` (2 s) mantêm o timer.
- Para consumir eventos (Módulos 3 e 4): `app.events.subscribe(ouvinte)`, onde o ouvinte tem `handle(event, evidence)`.

## Contrato do Módulo 1

`VisionPipeline.stream()` gera um `FrameAnalysis` por frame. Cada `PersonAnalysis` traz
`track_id`, `status`, `zone_id` e `missing_ppe`. O temporizador deve usar `frame.timestamp`,
que funciona tanto ao vivo quanto em vídeo gravado.
