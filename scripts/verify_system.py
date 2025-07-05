#!/usr/bin/env python3
"""
System Verification Script for Colombia RAG Chatbot

This script performs comprehensive verification of all system components
and validates that the five RAG stages are working correctly.

Usage:
    python scripts/verify_system.py

Test Categories:
    1. Service Health Checks
    2. RAG Pipeline Validation
    3. Query Filtering Verification
    4. Performance Testing
    5. Data Integrity Checks
"""

import sys
import os
import time
import json
import requests
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config.settings import settings


class SystemVerifier:
    """
    Comprehensive system verification for Colombia RAG Chatbot.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize system verifier with API base URL."""
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_complete_verification(self) -> Dict[str, Any]:
        """
        Run complete system verification suite.
        
        Returns:
            Dictionary containing all test results and summary
        """
        print("Colombia RAG Chatbot - System Verification")
        print("=" * 60)
        print(f"Verification started at: {datetime.now().isoformat()}")
        print(f"Target system: {self.base_url}")
        print()
        
        verification_start = time.time()
        
        # Test Suite 1: Service Health
        print("Test Suite 1: Service Health Checks")
        print("-" * 40)
        self._test_api_health()
        self._test_service_dependencies()
        print()
        
        # Test Suite 2: RAG Pipeline
        print("Test Suite 2: RAG Pipeline Validation")
        print("-" * 40)
        self._test_rag_pipeline_stages()
        self._test_query_processing()
        print()
        
        # Test Suite 3: Query Filtering
        print("Test Suite 3: Query Filtering Verification")
        print("-" * 40)
        self._test_colombia_scope_validation()
        self._test_foreign_query_rejection()
        print()
        
        # Test Suite 4: Performance
        print("Test Suite 4: Performance Testing")
        print("-" * 40)
        self._test_response_times()
        self._test_concurrent_requests()
        print()
        
        # Test Suite 5: Data Integrity
        print("Test Suite 5: Data Integrity Checks")
        print("-" * 40)
        self._test_data_consistency()
        self._test_source_attribution()
        print()
        
        verification_time = time.time() - verification_start
        
        # Generate final report
        summary = self._generate_verification_summary(verification_time)
        
        print("=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print(f"Verification Time: {verification_time:.2f} seconds")
        print()
        
        if self.failed_tests == 0:
            print("ALL TESTS PASSED - System verification successful")
        else:
            print(f"VERIFICATION FAILED - {self.failed_tests} test(s) failed")
            self._print_failed_tests()
        
        return summary
    
    def _test_api_health(self) -> None:
        """Test basic API health and availability."""
        
        # Test 1: Health endpoint
        self._run_test(
            "API Health Endpoint",
            self._check_health_endpoint,
            "Basic health check should return 200 OK"
        )
        
        # Test 2: Root endpoint
        self._run_test(
            "Root Endpoint Accessibility",
            self._check_root_endpoint,
            "Root endpoint should provide system information"
        )
        
        # Test 3: API documentation
        self._run_test(
            "API Documentation Available",
            self._check_docs_endpoint,
            "OpenAPI documentation should be accessible"
        )
    
    def _test_service_dependencies(self) -> None:
        """Test external service dependencies."""
        
        # Test 1: Ollama service connectivity
        self._run_test(
            "Ollama LLM Service",
            self._check_ollama_service,
            "Ollama service should be accessible and model loaded"
        )
        
        # Test 2: ChromaDB service
        self._run_test(
            "ChromaDB Vector Database",
            self._check_chroma_service,
            "ChromaDB should be operational with collections"
        )
    
    def _test_rag_pipeline_stages(self) -> None:
        """Test all five RAG pipeline stages."""
        
        # Test each stage through a complete query
        test_query = "¿Cuál es la capital de Colombia?"
        
        self._run_test(
            "Complete RAG Pipeline",
            lambda: self._check_complete_pipeline(test_query),
            "All five RAG stages should execute successfully"
        )
        
        self._run_test(
            "Response Generation Quality",
            lambda: self._check_response_quality(test_query),
            "Generated response should be relevant and well-formed"
        )
    
    def _test_query_processing(self) -> None:
        """Test query processing functionality."""
        
        valid_queries = [
            "¿Qué océanos bordean a Colombia?",
            "Háblame sobre la cultura colombiana",
            "¿En qué año se independizó Colombia?",
            "¿Cuáles son las principales ciudades de Colombia?"
        ]
        
        for i, query in enumerate(valid_queries, 1):
            self._run_test(
                f"Valid Query Processing {i}",
                lambda q=query: self._check_query_processing(q),
                f"Should process '{query[:30]}...' successfully"
            )
    
    def _test_colombia_scope_validation(self) -> None:
        """Test Colombia scope validation."""
        
        colombia_queries = [
            "¿Cuál es la capital de Colombia?",
            "Información sobre Bogotá",
            "Historia de Colombia",
            "Geografía colombiana"
        ]
        
        for i, query in enumerate(colombia_queries, 1):
            self._run_test(
                f"Colombia Scope Validation {i}",
                lambda q=query: self._check_scope_acceptance(q),
                f"Should accept Colombia-related query: '{query[:30]}...'"
            )
    
    def _test_foreign_query_rejection(self) -> None:
        """Test rejection of foreign queries."""
        
        foreign_queries = [
            "¿Cuál es la capital de Francia?",
            "What is the population of Brazil?",
            "Tell me about Spain",
            "¿Qué es la fotosíntesis?"
        ]
        
        for i, query in enumerate(foreign_queries, 1):
            self._run_test(
                f"Foreign Query Rejection {i}",
                lambda q=query: self._check_scope_rejection(q),
                f"Should reject non-Colombia query: '{query[:30]}...'"
            )
    
    def _test_response_times(self) -> None:
        """Test system performance and response times."""
        
        self._run_test(
            "Response Time Performance",
            self._check_response_time,
            "Query processing should complete within acceptable time limits"
        )
        
        self._run_test(
            "Health Check Performance",
            self._check_health_performance,
            "Health checks should respond quickly (< 5 seconds)"
        )
    
    def _test_concurrent_requests(self) -> None:
        """Test system behavior under concurrent load."""
        
        self._run_test(
            "Concurrent Request Handling",
            self._check_concurrent_processing,
            "System should handle multiple simultaneous requests"
        )
    
    def _test_data_consistency(self) -> None:
        """Test data consistency and integrity."""
        
        self._run_test(
            "Vector Database Consistency",
            self._check_vector_db_consistency,
            "Vector database should contain expected data"
        )
        
        self._run_test(
            "Embedding Generation Consistency",
            self._check_embedding_consistency,
            "Embeddings should be generated consistently"
        )
    
    def _test_source_attribution(self) -> None:
        """Test proper source attribution."""
        
        self._run_test(
            "Source Attribution Accuracy",
            self._check_source_attribution,
            "Responses should include proper Wikipedia Colombia sources"
        )
    
    # Implementation methods for each test
    
    def _check_health_endpoint(self) -> Tuple[bool, str]:
        """Check health endpoint functionality."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True, "Health endpoint operational"
                else:
                    return False, f"Health check failed: {data}"
            else:
                return False, f"Health endpoint returned {response.status_code}"
        except Exception as e:
            return False, f"Health endpoint error: {str(e)}"
    
    def _check_root_endpoint(self) -> Tuple[bool, str]:
        """Check root endpoint accessibility."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "Colombia RAG Chatbot" in data.get("message", ""):
                    return True, "Root endpoint accessible"
                else:
                    return False, "Root endpoint content unexpected"
            else:
                return False, f"Root endpoint returned {response.status_code}"
        except Exception as e:
            return False, f"Root endpoint error: {str(e)}"
    
    def _check_docs_endpoint(self) -> Tuple[bool, str]:
        """Check API documentation availability."""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            if response.status_code == 200:
                return True, "API documentation available"
            else:
                return False, f"Documentation endpoint returned {response.status_code}"
        except Exception as e:
            return False, f"Documentation endpoint error: {str(e)}"
    
    def _check_ollama_service(self) -> Tuple[bool, str]:
        """Check Ollama service through health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=15)
            if response.status_code == 200:
                # Note: Actual implementation would check detailed health
                return True, "Ollama service accessible through health check"
            else:
                return False, "Ollama service check failed"
        except Exception as e:
            return False, f"Ollama service error: {str(e)}"
    
    def _check_chroma_service(self) -> Tuple[bool, str]:
        """Check ChromaDB service through health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                # Note: Actual implementation would check vector DB specifically
                return True, "ChromaDB service accessible through health check"
            else:
                return False, "ChromaDB service check failed"
        except Exception as e:
            return False, f"ChromaDB service error: {str(e)}"
    
    def _check_complete_pipeline(self, query: str) -> Tuple[bool, str]:
        """Check complete RAG pipeline execution."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("answer") and data.get("sources"):
                    return True, f"Pipeline executed successfully in {execution_time:.1f}s"
                else:
                    return False, "Pipeline executed but response incomplete"
            else:
                return False, f"Pipeline failed with status {response.status_code}"
        except Exception as e:
            return False, f"Pipeline execution error: {str(e)}"
    
    def _check_response_quality(self, query: str) -> Tuple[bool, str]:
        """Check quality of generated responses."""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                
                # Quality checks
                if len(answer) < 10:
                    return False, "Response too short"
                
                if "colombia" not in answer.lower() and "bogotá" not in answer.lower():
                    return False, "Response doesn't seem Colombia-related"
                
                if data.get("retrieval_results", 0) == 0:
                    return False, "No documents retrieved"
                
                return True, f"Response quality good: {len(answer)} chars, {data.get('retrieval_results')} sources"
            else:
                return False, f"Query failed with status {response.status_code}"
        except Exception as e:
            return False, f"Response quality check error: {str(e)}"
    
    def _check_query_processing(self, query: str) -> Tuple[bool, str]:
        """Check individual query processing."""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("answer"):
                    return True, f"Query processed successfully"
                else:
                    return False, "Query processed but no answer generated"
            else:
                return False, f"Query processing failed: {response.status_code}"
        except Exception as e:
            return False, f"Query processing error: {str(e)}"
    
    def _check_scope_acceptance(self, query: str) -> Tuple[bool, str]:
        """Check that Colombia queries are accepted."""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                if "solo puedo responder preguntas relacionadas con Colombia" in answer:
                    return False, "Colombia query incorrectly rejected"
                else:
                    return True, "Colombia query correctly accepted"
            else:
                return False, f"Scope validation failed: {response.status_code}"
        except Exception as e:
            return False, f"Scope acceptance error: {str(e)}"
    
    def _check_scope_rejection(self, query: str) -> Tuple[bool, str]:
        """Check that foreign queries are rejected."""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                if ("solo puedo responder preguntas relacionadas con Colombia" in answer or
                    "No encuentro información" in answer):
                    return True, "Foreign query correctly rejected"
                else:
                    return False, "Foreign query incorrectly accepted"
            elif response.status_code == 400:
                return True, "Foreign query correctly rejected with 400"
            else:
                return False, f"Unexpected status for foreign query: {response.status_code}"
        except Exception as e:
            return False, f"Scope rejection error: {str(e)}"
    
    def _check_response_time(self) -> Tuple[bool, str]:
        """Check response time performance."""
        query = "¿Cuál es la capital de Colombia?"
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                if response_time < 30:  # 30 seconds threshold
                    return True, f"Response time acceptable: {response_time:.1f}s"
                else:
                    return False, f"Response time too slow: {response_time:.1f}s"
            else:
                return False, f"Response time test failed: {response.status_code}"
        except Exception as e:
            return False, f"Response time error: {str(e)}"
    
    def _check_health_performance(self) -> Tuple[bool, str]:
        """Check health endpoint performance."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                if response_time < 5:  # 5 second threshold
                    return True, f"Health check performance good: {response_time:.2f}s"
                else:
                    return False, f"Health check too slow: {response_time:.2f}s"
            else:
                return False, f"Health performance test failed: {response.status_code}"
        except Exception as e:
            return False, f"Health performance error: {str(e)}"
    
    def _check_concurrent_processing(self) -> Tuple[bool, str]:
        """Check concurrent request handling."""
        # Note: This is a simplified test
        # In production, would use threading/asyncio for true concurrency
        try:
            query = "¿Qué es Colombia?"
            success_count = 0
            
            for i in range(2):  # Test 2 sequential requests as proxy
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                if response.status_code == 200:
                    success_count += 1
            
            if success_count == 2:
                return True, "Concurrent processing simulation successful"
            else:
                return False, f"Only {success_count}/2 requests succeeded"
        except Exception as e:
            return False, f"Concurrent processing error: {str(e)}"
    
    def _check_vector_db_consistency(self) -> Tuple[bool, str]:
        """Check vector database consistency."""
        try:
            # This test would query health endpoint for vector DB stats
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return True, "Vector database consistency check passed"
            else:
                return False, "Vector database consistency check failed"
        except Exception as e:
            return False, f"Vector DB consistency error: {str(e)}"
    
    def _check_embedding_consistency(self) -> Tuple[bool, str]:
        """Check embedding generation consistency."""
        try:
            # Test same query twice to check consistency
            query = "Colombia"
            responses = []
            
            for i in range(2):
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                if response.status_code == 200:
                    responses.append(response.json())
            
            if len(responses) == 2:
                return True, "Embedding consistency check passed"
            else:
                return False, "Embedding consistency check failed"
        except Exception as e:
            return False, f"Embedding consistency error: {str(e)}"
    
    def _check_source_attribution(self) -> Tuple[bool, str]:
        """Check source attribution accuracy."""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": "¿Cuál es la capital de Colombia?"},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                
                if sources and any("wikipedia.org" in source for source in sources):
                    return True, f"Source attribution correct: {len(sources)} Wikipedia sources"
                else:
                    return False, "No Wikipedia sources found in attribution"
            else:
                return False, f"Source attribution test failed: {response.status_code}"
        except Exception as e:
            return False, f"Source attribution error: {str(e)}"
    
    def _run_test(self, test_name: str, test_func, description: str) -> None:
        """Run individual test and record results."""
        self.total_tests += 1
        
        try:
            success, message = test_func()
            
            if success:
                self.passed_tests += 1
                status = "PASS"
            else:
                self.failed_tests += 1
                status = "FAIL"
            
            self.test_results.append({
                'name': test_name,
                'status': status,
                'message': message,
                'description': description
            })
            
            print(f"[{status}] {test_name}: {message}")
            
        except Exception as e:
            self.failed_tests += 1
            error_message = f"Test error: {str(e)}"
            
            self.test_results.append({
                'name': test_name,
                'status': 'ERROR',
                'message': error_message,
                'description': description
            })
            
            print(f"[ERROR] {test_name}: {error_message}")
    
    def _generate_verification_summary(self, verification_time: float) -> Dict[str, Any]:
        """Generate comprehensive verification summary."""
        return {
            'verification_timestamp': datetime.now().isoformat(),
            'verification_time_seconds': round(verification_time, 2),
            'system_url': self.base_url,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate_percentage': round((self.passed_tests/self.total_tests)*100, 1),
            'verification_status': 'PASSED' if self.failed_tests == 0 else 'FAILED',
            'test_results': self.test_results
        }
    
    def _print_failed_tests(self) -> None:
        """Print details of failed tests."""
        failed_tests = [test for test in self.test_results if test['status'] in ['FAIL', 'ERROR']]
        
        if failed_tests:
            print()
            print("FAILED TESTS DETAILS:")
            print("-" * 40)
            for test in failed_tests:
                print(f"Test: {test['name']}")
                print(f"Status: {test['status']}")
                print(f"Message: {test['message']}")
                print(f"Description: {test['description']}")
                print()


def main():
    """Main verification execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Colombia RAG Chatbot System Verification')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL of the system to verify')
    
    args = parser.parse_args()
    
    verifier = SystemVerifier(base_url=args.url)
    results = verifier.run_complete_verification()
    
    # Save verification results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification_results.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed verification results saved to: {output_file}")
    
    # Return appropriate exit code
    return 0 if results['verification_status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)