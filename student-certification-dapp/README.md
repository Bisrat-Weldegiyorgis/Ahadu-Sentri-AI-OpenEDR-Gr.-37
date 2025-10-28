# Decentralized Student Certification DApp

A complete web3 DApp that lets a verified institution issue blockchain-backed student certificates on Ethereum (Sepolia). Students (or verifiers) can connect a wallet and verify the certificate using the student address.

- **Frontend**: React + Vite + TailwindCSS
- **Blockchain lib**: ethers.js v6
- **Smart Contract**: Solidity 0.8.18
- **Network**: Sepolia (Ethereum testnet)
- **Build/Deploy**: Hardhat (optional), or Remix

## How it works

- The contract `StudentCertificate` stores a certificate per student address.
- Only the deployer (issuer) can call `issueCertificate`.
- Anyone can call `getCertificate(address)` to verify a student's certificate.
- The frontend allows:
  - Connecting MetaMask
  - Setting the deployed contract address
  - Issuer-only UI to issue certificates
  - Public verification by address

## Project Structure

```
student-certification-dapp/
├─ contracts/
│  └─ StudentCertificate.sol
├─ scripts/
│  └─ deploy.js
├─ src/
│  ├─ App.jsx
│  ├─ main.jsx
│  ├─ index.css
│  └─ abi.json
├─ .env.example
├─ hardhat.config.js
├─ index.html
├─ package.json
├─ postcss.config.js
├─ tailwind.config.js
├─ vite.config.mjs
└─ README.md
```

## Prerequisites

- Node.js 18+
- MetaMask browser extension
- A Sepolia RPC URL (e.g., from Alchemy/Infura) and a funded Sepolia account for deployment

## Getting Started (Local)

```bash
# 1) Install dependencies
npm install

# 2) Run the dev server
npm run dev
```

Open `http://localhost:5173` in your browser.

## Deploying the Contract

You can deploy using Remix or Hardhat.

### Option A: Deploy with Remix

1. Open Remix at `https://remix.ethereum.org`.
2. Create a new file `StudentCertificate.sol` and paste the contents of `contracts/StudentCertificate.sol`.
3. In the "Solidity Compiler" tab, select compiler `0.8.18`, compile.
4. In the "Deploy & Run" tab:
   - Environment: Injected Provider (MetaMask)
   - Network: Sepolia
   - Click "Deploy"
5. Copy the deployed contract address and paste it into the app UI, then click "Save".

### Option B: Deploy with Hardhat

1. Copy `.env.example` to `.env` and fill in values:

```
SEPOLIA_RPC_URL=YOUR_RPC_URL
PRIVATE_KEY=YOUR_PRIVATE_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
```

2. Compile and deploy:

```bash
npm run compile
npm run deploy:sepolia
```

3. Note the contract address from the output and paste it into the app UI.

Optionally verify on Etherscan (if API key set):

```bash
npx hardhat verify --network sepolia YOUR_DEPLOYED_ADDRESS
```

## Using the DApp

- Click "Connect Wallet" to connect MetaMask.
- Paste your deployed contract address and click "Save".
- If you are the deployer (issuer), the "Issue Certificate" section appears.
- To verify, enter a student address and click "Lookup". The certificate details, if issued, will be shown.

## Security Considerations

- **Access control**: `onlyIssuer` modifier enforces that only the contract deployer can issue certificates.
- **Immutability**: Once deployed, the `issuer` is immutable, and issued certificates are stored on chain.
- **No reentrancy risk**: No external calls or payable functions; state changes are straightforward.
- **Input validation**: Student address, name, course, and year are validated.
- **Transparency**: Anyone can read a certificate via `getCertificate`.

## Notes on ethers.js v6 Integration

- Uses `BrowserProvider` and `getSigner()` for MetaMask.
- Contract is instantiated with either `signer` (for writes) or `provider` (for reads).
- BigInt is used for numeric values (e.g., the year).

## Screenshots to Capture for Submission

- Wallet connection popup and in-app connected state
- Contract deployment transaction on Sepolia
- Issuing a certificate transaction
- Successful verification showing certificate data in the UI

## Troubleshooting

- If the Issuer form does not appear, ensure your wallet is the deployer address.
- Ensure the contract address is correct and on Sepolia.
- If you see "Invalid address", double check the input format.
- Clear `localStorage` entry `STUDENT_CERT_CONTRACT_ADDRESS` if needed and re-enter.

## License

MIT
