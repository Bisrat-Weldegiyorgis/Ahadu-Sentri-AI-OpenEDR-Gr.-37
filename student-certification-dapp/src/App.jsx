import React, { useEffect, useMemo, useState } from 'react'
import { ethers } from 'ethers'
import abi from './abi.json'

const CONTRACT_ADDRESS_KEY = 'STUDENT_CERT_CONTRACT_ADDRESS'

function shorten(address) {
  return address ? address.slice(0, 6) + '…' + address.slice(-4) : ''
}

export default function App() {
  const [provider, setProvider] = useState(null)
  const [signer, setSigner] = useState(null)
  const [account, setAccount] = useState(null)
  const [isIssuer, setIsIssuer] = useState(false)
  const [status, setStatus] = useState('')
  const [contractAddress, setContractAddress] = useState(
    localStorage.getItem(CONTRACT_ADDRESS_KEY) || ''
  )
  const [studentAddress, setStudentAddress] = useState('')
  const [studentName, setStudentName] = useState('')
  const [course, setCourse] = useState('')
  const [issueYear, setIssueYear] = useState('2025')

  const contract = useMemo(() => {
    if (!contractAddress || !(signer || provider)) return null
    const runner = signer || provider
    return new ethers.Contract(contractAddress, abi, runner)
  }, [contractAddress, signer, provider])

  // Initialize provider
  useEffect(() => {
    if (typeof window.ethereum !== 'undefined') {
      const newProvider = new ethers.BrowserProvider(window.ethereum)
      setProvider(newProvider)
    }
  }, [])

  // Handle account changes
  useEffect(() => {
    if (!provider) return
    provider.send('eth_accounts', []).then(async (accounts) => {
      if (accounts.length > 0) {
        setAccount(ethers.getAddress(accounts[0]))
        const s = await provider.getSigner()
        setSigner(s)
      }
    })

    if (window.ethereum && window.ethereum.on) {
      const handleAccountsChanged = async (accounts) => {
        if (accounts.length === 0) {
          setAccount(null)
          setSigner(null)
        } else {
          setAccount(ethers.getAddress(accounts[0]))
          const s = await provider.getSigner()
          setSigner(s)
        }
      }
      window.ethereum.on('accountsChanged', handleAccountsChanged)
      return () => window.ethereum.removeListener('accountsChanged', handleAccountsChanged)
    }
  }, [provider])

  // Determine if current account is issuer
  useEffect(() => {
    (async () => {
      if (!contract || !account) return
      try {
        const issuer = await contract.issuer()
        setIsIssuer(issuer.toLowerCase() === account.toLowerCase())
      } catch {}
    })()
  }, [contract, account])

  const connectWallet = async () => {
    if (!provider) {
      setStatus('Please install MetaMask.')
      return
    }
    try {
      const accounts = await provider.send('eth_requestAccounts', [])
      if (accounts.length > 0) {
        setAccount(ethers.getAddress(accounts[0]))
        const s = await provider.getSigner()
        setSigner(s)
      }
    } catch (err) {
      setStatus(err.message || 'Wallet connection rejected')
    }
  }

  const saveContractAddress = () => {
    try {
      if (!ethers.isAddress(contractAddress)) {
        setStatus('Invalid contract address')
        return
      }
      localStorage.setItem(CONTRACT_ADDRESS_KEY, contractAddress)
      setStatus('Contract address saved')
    } catch (e) {
      setStatus('Failed to save contract address')
    }
  }

  const handleIssue = async (e) => {
    e.preventDefault()
    if (!contract || !signer) return setStatus('Connect wallet and set contract')
    if (!ethers.isAddress(studentAddress)) return setStatus('Invalid student address')
    try {
      const connected = contract.connect(signer)
      const tx = await connected.issueCertificate(
        studentAddress,
        studentName,
        course,
        BigInt(issueYear)
      )
      setStatus('Issuing… waiting for confirmation: ' + tx.hash)
      await tx.wait()
      setStatus('Certificate issued! Tx: ' + tx.hash)
    } catch (err) {
      setStatus(err.shortMessage || err.message)
    }
  }

  const [lookupAddress, setLookupAddress] = useState('')
  const [lookupResult, setLookupResult] = useState(null)

  const handleLookup = async (e) => {
    e.preventDefault()
    if (!contract) return setStatus('Set contract address first')
    if (!ethers.isAddress(lookupAddress)) return setStatus('Invalid address')
    try {
      const cert = await contract.getCertificate(lookupAddress)
      if (!cert || cert.issued === false) {
        setLookupResult(null)
        setStatus('No certificate for this address')
      } else {
        setLookupResult(cert)
        setStatus('Certificate found')
      }
    } catch (err) {
      setStatus(err.shortMessage || err.message)
    }
  }

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Decentralized Student Certification DApp</h1>
          <button
            onClick={account ? undefined : connectWallet}
            className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500"
          >
            {account ? `Connected: ${shorten(account)}` : 'Connect Wallet'}
          </button>
        </header>

        <section className="bg-gray-900 border border-gray-800 rounded p-4">
          <h2 className="font-semibold mb-2">Contract Address</h2>
          <div className="flex gap-2">
            <input
              className="flex-1 px-3 py-2 rounded bg-gray-800 border border-gray-700"
              placeholder="0x..."
              value={contractAddress}
              onChange={(e) => setContractAddress(e.target.value)}
            />
            <button onClick={saveContractAddress} className="px-3 py-2 rounded bg-slate-700 hover:bg-slate-600">Save</button>
          </div>
          <p className="text-sm text-gray-400 mt-1">This should be your deployed contract on Sepolia.</p>
        </section>

        {isIssuer && (
          <section className="bg-gray-900 border border-gray-800 rounded p-4">
            <h2 className="font-semibold mb-4">Issuer: Issue Certificate</h2>
            <form onSubmit={handleIssue} className="grid gap-3 md:grid-cols-2">
              <input
                className="px-3 py-2 rounded bg-gray-800 border border-gray-700"
                placeholder="Student Address (0x...)"
                value={studentAddress}
                onChange={(e) => setStudentAddress(e.target.value)}
              />
              <input
                className="px-3 py-2 rounded bg-gray-800 border border-gray-700"
                placeholder="Student Name"
                value={studentName}
                onChange={(e) => setStudentName(e.target.value)}
              />
              <input
                className="px-3 py-2 rounded bg-gray-800 border border-gray-700"
                placeholder="Course/Program"
                value={course}
                onChange={(e) => setCourse(e.target.value)}
              />
              <input
                className="px-3 py-2 rounded bg-gray-800 border border-gray-700"
                placeholder="Year"
                type="number"
                value={issueYear}
                onChange={(e) => setIssueYear(e.target.value)}
              />
              <div className="md:col-span-2">
                <button className="w-full px-4 py-2 rounded bg-green-600 hover:bg-green-500">Issue Certificate</button>
              </div>
            </form>
          </section>
        )}

        <section className="bg-gray-900 border border-gray-800 rounded p-4">
          <h2 className="font-semibold mb-4">Verify Certificate</h2>
          <form onSubmit={handleLookup} className="flex gap-2">
            <input
              className="flex-1 px-3 py-2 rounded bg-gray-800 border border-gray-700"
              placeholder="Student Address (0x...)"
              value={lookupAddress}
              onChange={(e) => setLookupAddress(e.target.value)}
            />
            <button className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500">Lookup</button>
          </form>

          {lookupResult && (
            <div className="mt-4 grid gap-2">
              <div className="text-sm text-gray-400">Student: {lookupResult.student}</div>
              <div className="text-sm text-gray-400">Name: {lookupResult.name}</div>
              <div className="text-sm text-gray-400">Course: {lookupResult.course}</div>
              <div className="text-sm text-gray-400">Year: {lookupResult.year?.toString?.() ?? lookupResult.year}</div>
              <div className="text-sm text-gray-400">Issued: {lookupResult.issued ? 'Yes' : 'No'}</div>
            </div>
          )}
        </section>

        <section className="bg-gray-900 border border-gray-800 rounded p-4">
          <h2 className="font-semibold mb-2">Status</h2>
          <div className="text-sm text-gray-300 break-words min-h-[1.5rem]">{status}</div>
        </section>

        <footer className="text-center text-xs text-gray-500">
          Built for Blockchain Talent Development Curriculum — Sepolia, Solidity, ethers v6
        </footer>
      </div>
    </div>
  )
}
